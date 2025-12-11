#!/usr/bin/env python3
"""
Simulador de Cache Hiper-Virtualizado (CORRIGIDO)
Correções aplicadas:
1. Mudança de threading.Lock para threading.RLock em CacheBase para evitar deadlock
   quando o Hypervisor e o Cache tentam adquirir o mesmo lock na mesma thread.
2. Adição de try/finally no process_requests da VM para garantir que task_done()
   seja sempre chamado, evitando travamento da queue.join().
"""

import sys
import json
import csv
import random
import threading
import time
from queue import Queue, Empty
from collections import OrderedDict, defaultdict, deque
from typing import Optional, List, Dict, Tuple

# ---------- Configurável: ativar sleep para latência realista (False por padrão) ----------
SIMULATE_SLEEP = False


class CacheBase:
    def __init__(self, capacity: int, policy: str):
        self.capacity = max(0, int(capacity))
        self.policy = policy.upper()
        self.evictions = 0
        self.insertions = 0
        # CORREÇÃO CRÍTICA: RLock permite que a mesma thread (Hypervisor) 
        # adquira o lock novamente dentro dos métodos do Cache (get/put)
        self.lock = threading.RLock() 

    def get(self, key: str) -> bool:
        raise NotImplementedError

    def put(self, key: str):
        raise NotImplementedError

    def info(self) -> Dict:
        with self.lock:
            return {"capacity": self.capacity, "policy": self.policy,
                    "evictions": self.evictions, "insertions": self.insertions}


class LRUCache(CacheBase):
    def __init__(self, capacity: int):
        super().__init__(capacity, "LRU")
        self.od = OrderedDict()

    def get(self, key):
        with self.lock:
            if key in self.od:
                self.od.move_to_end(key)
                return True
            return False

    def put(self, key):
        with self.lock:
            if self.capacity == 0:
                return
            if key in self.od:
                self.od.move_to_end(key)
                return
            if len(self.od) >= self.capacity:
                self.od.popitem(last=False)
                self.evictions += 1
            self.od[key] = True
            self.insertions += 1


class FIFOCache(CacheBase):
    def __init__(self, capacity: int):
        super().__init__(capacity, "FIFO")
        self.queue = deque()
        self.set = set()

    def get(self, key):
        with self.lock:
            return key in self.set

    def put(self, key):
        with self.lock:
            if self.capacity == 0:
                return
            if key in self.set:
                return
            if len(self.queue) >= self.capacity:
                old = self.queue.popleft()
                self.set.remove(old)
                self.evictions += 1
            self.queue.append(key)
            self.set.add(key)
            self.insertions += 1


class LFUCache(CacheBase):
    def __init__(self, capacity: int):
        super().__init__(capacity, "LFU")
        self.freq = defaultdict(int)
        self.storage = set()
        self.timer = 0
        self.last_access = {}

    def get(self, key):
        with self.lock:
            if key in self.storage:
                self.freq[key] += 1
                self.timer += 1
                self.last_access[key] = self.timer
                return True
            return False

    def put(self, key):
        with self.lock:
            if self.capacity == 0:
                return

            if key in self.storage:
                self.freq[key] += 1
                self.timer += 1
                self.last_access[key] = self.timer
                return

            if len(self.storage) >= self.capacity:
                # Empata por última vez acessada para desempate estável
                victim = min(self.storage, key=lambda k: (self.freq[k], self.last_access[k]))
                self.storage.remove(victim)
                del self.freq[victim]
                del self.last_access[victim]
                self.evictions += 1

            self.storage.add(key)
            self.freq[key] = 1
            self.timer += 1
            self.last_access[key] = self.timer
            self.insertions += 1


def make_cache(capacity: int, policy: str) -> CacheBase:
    p = policy.upper()
    if p == "LRU":
        return LRUCache(capacity)
    elif p == "FIFO":
        return FIFOCache(capacity)
    elif p == "LFU":
        return LFUCache(capacity)
    else:
        raise ValueError(f"Unknown cache policy: {policy}")


class Disk:
    def __init__(self, latency: int):
        self.latency = int(latency)
        self.lock = threading.Lock()
        self.access_count = 0

    def fetch(self, file_id: str) -> Tuple[str, int]:
        with self.lock:
            self.access_count += 1
        if SIMULATE_SLEEP:
            time.sleep(self.latency / 1000.0)
        return (f"DATA({file_id})", self.latency)


class Hypervisor:
    def __init__(self, host_cache: CacheBase, host_latency: int, disk: Disk):
        self.host_cache = host_cache
        self.host_latency = int(host_latency)
        self.disk = disk
        self.host_hits = 0
        self.disk_fetches = 0
        self.lock = threading.Lock()  # Protege contadores
        self.contention_time = 0.0  # Tempo esperando por locks (segundos)
        self.lock_waits = 0

    def fetch(self, file_id: str) -> Tuple[str, int]:
        """
        Retorna (where, latency)
        where in {"host", "disk"}
        """
        lock_start = time.time()
        
        # Aqui o Hypervisor pega o lock do cache
        self.host_cache.lock.acquire()
        
        lock_wait = time.time() - lock_start

        # contabiliza contenção somente se houve espera significativa (> 1 microseg)
        if lock_wait > 1e-6:
            with self.lock:
                self.lock_waits += 1
                self.contention_time += lock_wait

        try:
            # Como agora é RLock, essa chamada interna (que faz 'with lock') não trava mais
            if self.host_cache.get(file_id):
                with self.lock:
                    self.host_hits += 1
                if SIMULATE_SLEEP:
                    time.sleep(self.host_latency / 1000.0)
                return ("host", self.host_latency)

            # Miss: busca no disco e insere no host cache
            content, dlat = self.disk.fetch(file_id)
            self.host_cache.put(file_id)
            with self.lock:
                self.disk_fetches += 1
            return ("disk", dlat)
        finally:
            # Garantir release do lock do cache do host
            try:
                self.host_cache.lock.release()
            except RuntimeError:
                print("[WARN] host_cache.lock.release() falhou - provavelmente já liberado")


class VM:
    def __init__(self, vm_id: int, vm_cache: CacheBase, vm_latency: int, hypervisor: Hypervisor):
        self.vm_id = vm_id
        self.cache = vm_cache
        self.vm_latency = int(vm_latency)
        self.hypervisor = hypervisor
        self.accesses = 0
        self.vm_hits = 0
        self.host_hits = 0
        self.disk_hits = 0
        self.latency = 0.0
        self.lock = threading.Lock()

        # Para execução concorrente
        self.thread: Optional[threading.Thread] = None
        self.request_queue: Queue = Queue()
        self.access_log: List[Dict] = []

    def access(self, file_id: str, promote_to_vm: bool = True) -> Tuple[str, int]:
        """
        Retorna (where, latency)
        where: "vm", "host" ou "disk"
        latency: tempo em ms associado ao acesso (somente a latência específica)
        """
        with self.lock:
            self.accesses += 1

        # Primeiro tenta no cache da VM
        if self.cache.get(file_id):
            with self.lock:
                self.vm_hits += 1
                self.latency += self.vm_latency
            if SIMULATE_SLEEP:
                time.sleep(self.vm_latency / 1000.0)
            return ("vm", self.vm_latency)

        # Caso falta na VM, vai ao hypervisor
        where, latency = self.hypervisor.fetch(file_id)
        total_latency = latency + self.vm_latency

        with self.lock:
            if where == "host":
                self.host_hits += 1
            else:
                self.disk_hits += 1

            if promote_to_vm:
                # Promove o bloco para cache da VM
                self.cache.put(file_id)

            self.latency += total_latency

        if SIMULATE_SLEEP:
            # simula latência da VM também
            time.sleep(self.vm_latency / 1000.0)

        return (where, total_latency)

    def process_requests(self):
        thread_name = threading.current_thread().name

        while True:
            # Espera até receber uma tarefa
            step, file_id = self.request_queue.get()

            try:
                # Sentinel → terminar a thread
                if step is None and file_id is None:
                    break
                
                # Processa acesso
                where, lat = self.access(file_id)
                self.access_log.append({
                    "step": step,
                    "vm": self.vm_id,
                    "file": file_id,
                    "where": where,
                    "latency": lat,
                    "vm_cache_size": self.cache.capacity,
                    "host_cache_size": self.hypervisor.host_cache.capacity,
                    "thread_id": thread_name
                })
            except Exception as e:
                print(f"[ERRO] VM-{self.vm_id} falhou ao processar arquivo {file_id}: {e}")
            finally:
                # CORREÇÃO: task_done deve ser chamado no finally para evitar travamento da queue
                self.request_queue.task_done()

    def start_concurrent(self):
        """Inicia thread da VM"""
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(
            target=self.process_requests,
            name=f"VM-{self.vm_id}",
            daemon=False
        )
        self.thread.start()

    def stop_and_join(self, timeout: float = 5.0):
        """Garante que a thread será finalizada"""
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=timeout)
            if self.thread.is_alive():
                print(f"[AVISO] Thread VM-{self.vm_id} não finalizou após {timeout}s")

    def enqueue_access(self, step: Optional[int], file_id: Optional[str]):
        """Adiciona requisição na fila"""
        self.request_queue.put((step, file_id))

    def stats(self) -> Dict:
        with self.lock:
            return {
                "vm_id": self.vm_id,
                "accesses": self.accesses,
                "vm_hits": self.vm_hits,
                "host_hits": self.host_hits,
                "disk_hits": self.disk_hits,
                "latency": self.latency,
                "vm_cache_info": self.cache.info()
            }


class Simulator:
    def __init__(self, cfg: Dict):
        random.seed(cfg.get("seed", 0))
        self.cfg = cfg
        self.execution_mode = cfg.get("execution_mode", "sequential")

        self.disk = Disk(cfg["disk_latency"])
        self.host_cache = make_cache(cfg["host_cache_size"], cfg["host_cache_policy"])
        self.hypervisor = Hypervisor(self.host_cache, cfg["host_latency"], self.disk)
        self.vms: List[VM] = []

        for vm_id in range(cfg["vm_count"]):
            vm_cache = make_cache(cfg["vm_cache_size"], cfg["vm_cache_policy"])
            vm = VM(vm_id, vm_cache, cfg["vm_latency"], self.hypervisor)
            self.vms.append(vm)

        self.workload = self._build_workload(cfg["workload"])
        self.access_log: List[Dict] = []

    def _build_workload(self, wcfg: Dict) -> List[Dict]:
        mode = wcfg.get("mode", "random")
        if mode == "provided":
            return wcfg.get("accesses", [])
        elif mode == "random":
            rnd = wcfg.get("random", {})
            length = rnd.get("length", 100)
            files = rnd.get("files", ["A", "B", "C", "D", "E", "F", "G"])
            res = []
            for step in range(length):
                vm = random.randrange(self.cfg["vm_count"])
                file_id = random.choice(files)
                res.append({"step": step, "vm": vm, "file": file_id})
            return res
        else:
            raise ValueError("Unknown workload mode")

    def run_sequential(self):
        """Execução sequencial (um acesso por vez)"""
        print(f"[SIMULADOR] Modo: SEQUENCIAL")
        for op in self.workload:
            step = op.get("step")
            vm_id = int(op["vm"])
            file_id = str(op["file"])
            vm = self.vms[vm_id]
            where, lat = vm.access(file_id)
            self.access_log.append({
                "step": step,
                "vm": vm_id,
                "file": file_id,
                "where": where,
                "latency": lat,
                "vm_cache_size": vm.cache.capacity,
                "host_cache_size": self.host_cache.capacity,
                "thread_id": None
            })

    def run_concurrent(self):
        """
        Execução concorrente
        """
        print(f"[SIMULADOR] Modo: CONCORRENTE ({len(self.vms)} VMs em paralelo)")

        # 1) Inicia workers
        for vm in self.vms:
            vm.start_concurrent()

        # 2) Distribui workload para as VMs
        print(f"[SIMULADOR] Distribuindo {len(self.workload)} acessos.")
        for op in self.workload:
            step = op.get("step")
            vm_id = int(op["vm"])
            file_id = str(op["file"])
            self.vms[vm_id].enqueue_access(step, file_id)

        # 3) Envia sentinels: um para cada VM
        for vm in self.vms:
            vm.enqueue_access(None, None)

        print(f"[SIMULADOR] Aguardando processamento (queue.join)...")

        # 4) Aguarda todas as filas serem processadas
        for vm in self.vms:
            vm.request_queue.join()

        # 5) Aguarda threads finalizarem
        for vm in self.vms:
            vm.stop_and_join(timeout=5.0)

        print(f"[SIMULADOR] Consolidando logs.")

        # Consolida logs de todas as VMs
        for vm in self.vms:
            self.access_log.extend(vm.access_log)

        # Ordena por step para manter sequência temporal
        self.access_log.sort(key=lambda x: (x["step"] if x["step"] is not None else -1, x["vm"]))

        print(f"[SIMULADOR] Contenção detectada: {self.hypervisor.lock_waits} esperas por lock")
        print(f"[SIMULADOR] Tempo total em contenção: {self.hypervisor.contention_time*1000:.2f}ms")

    def run(self):
        """Executa simulação no modo configurado"""
        start_time = time.time()

        if self.execution_mode == "concurrent":
            self.run_concurrent()
        else:
            self.run_sequential()

        elapsed = time.time() - start_time
        print(f"[SIMULADOR] Tempo de execução: {elapsed:.3f}s")

    def collect_results(self) -> Dict:
        vms_stats = [vm.stats() for vm in self.vms]
        host_info = self.hypervisor.host_cache.info()
        host_metrics = {
            "host_hits": self.hypervisor.host_hits,
            "disk_fetches": self.hypervisor.disk_fetches,
            "host_cache_info": host_info,
            "lock_contention": {
                "lock_waits": self.hypervisor.lock_waits,
                "total_contention_time_ms": self.hypervisor.contention_time * 1000
            }
        }
        totals = {
            "total_accesses": len(self.workload),
            "total_latency": sum(a["latency"] for a in self.access_log),
            "execution_mode": self.execution_mode
        }
        return {"vms": vms_stats, "host": host_metrics, "totals": totals, "access_log": self.access_log}

    def save_outputs(self, json_path: Optional[str], csv_path: Optional[str]):
        results = self.collect_results()
        if json_path:
            with open(json_path, "w") as f:
                json.dump(results, f, indent=4)
        if csv_path:
            with open(csv_path, "w", newline="") as f:
                fieldnames = ["step", "vm", "file", "where", "latency", "vm_cache_size", "host_cache_size"]
                if self.execution_mode == "concurrent":
                    fieldnames.append("thread_id")
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in results["access_log"]:
                    # padroniza campos ausentes
                    row_out = {k: row.get(k, "") for k in fieldnames}
                    writer.writerow(row_out)


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def main():
    if len(sys.argv) < 2:
        print("Uso: python3 cache_virtual.py config.json")
        sys.exit(1)
    cfg = load_config(sys.argv[1])
    sim = Simulator(cfg)
    sim.run()
    sim.save_outputs(cfg.get("output_json", "results.json"), cfg.get("output_csv", "results.csv"))


if __name__ == "__main__":
    main()