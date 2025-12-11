import sys
import json
import csv
import random
from collections import OrderedDict, defaultdict, deque
from typing import Optional, List, Dict, Tuple

class CacheBase:
    def __init__(self, capacity: int, policy: str):
        self.capacity = max(0, int(capacity))
        self.policy = policy.upper()
        self.evictions = 0
        self.insertions = 0

    def get(self, key: str) -> bool:
        raise NotImplementedError

    def put(self, key: str):
        raise NotImplementedError

    def info(self) -> Dict:
        return {"capacity": self.capacity, "policy": self.policy,
                "evictions": self.evictions, "insertions": self.insertions}

class LRUCache(CacheBase):
    def __init__(self, capacity: int):
        super().__init__(capacity, "LRU")
        self.od = OrderedDict()

    def get(self, key):
        if key in self.od:
            self.od.move_to_end(key)
            return True
        return False

    def put(self, key):
        if self.capacity == 0: return
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
        return key in self.set

    def put(self, key):
        if self.capacity == 0: return
        if key in self.set: return
        if len(self.queue) >= self.capacity:
            old = self.queue.popleft()
            self.set.remove(old)
            self.evictions += 1
        self.queue.append(key)
        self.set.add(key)
        self.insertions += 1

class LFUCache(CacheBase):
    """
    LFU CORRIGIDO: Usa um contador lógico (timer) para desempatar frequências iguais.
    Remove o item com menor frequência e, em caso de empate, o mais antigo (LRU logic).
    """
    def __init__(self, capacity: int):
        super().__init__(capacity, "LFU")
        self.freq = defaultdict(int)
        self.storage = set()
        self.timer = 0
        self.last_access = {} 

    def get(self, key):
        if key in self.storage:
            self.freq[key] += 1
            self.timer += 1
            self.last_access[key] = self.timer
            return True
        return False

    def put(self, key):
        if self.capacity == 0: return
        
        # Se já existe, atualiza
        if key in self.storage:
            self.freq[key] += 1
            self.timer += 1
            self.last_access[key] = self.timer
            return
        
        # Se precisa inserir novo e está cheio
        if len(self.storage) >= self.capacity:
            # Critério de remoção: Menor Frequência -> Menor Last Access (Mais antigo)
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
    if p == "LRU": return LRUCache(capacity)
    elif p == "FIFO": return FIFOCache(capacity)
    elif p == "LFU": return LFUCache(capacity)
    else: raise ValueError(f"Unknown cache policy: {policy}")

class Disk:
    def __init__(self, latency: int):
        self.latency = int(latency)

    def fetch(self, file_id: str) -> Tuple[str, int]:
        return (f"DATA({file_id})", self.latency)

class Hypervisor:
    def __init__(self, host_cache: CacheBase, host_latency: int, disk: Disk):
        self.host_cache = host_cache
        self.host_latency = int(host_latency)
        self.disk = disk
        self.host_hits = 0
        self.disk_fetches = 0

    def fetch(self, file_id: str) -> Tuple[str, str, int]:
        if self.host_cache.get(file_id):
            self.host_hits += 1
            return ("host", f"DATA({file_id})", self.host_latency)
        content, dlat = self.disk.fetch(file_id)
        self.host_cache.put(file_id)
        self.disk_fetches += 1
        return ("disk", content, dlat)

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
        self.latency = 0

    def access(self, file_id: str, promote_to_vm: bool = True):
        self.accesses += 1
        if self.cache.get(file_id):
            self.vm_hits += 1
            self.latency += self.vm_latency
            return ("vm", self.vm_latency)

        where, content, latency = self.hypervisor.fetch(file_id)
        total_latency = latency + self.vm_latency
        if where == "host": self.host_hits += 1
        else: self.disk_hits += 1

        if promote_to_vm:
            self.cache.put(file_id)

        self.latency += total_latency
        return (where, total_latency)

    def stats(self) -> Dict:
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
        self.disk = Disk(cfg["disk_latency"])
        self.host_cache = make_cache(cfg["host_cache_size"], cfg["host_cache_policy"])
        self.hypervisor = Hypervisor(self.host_cache, cfg["host_latency"], self.disk)
        self.vms: List[VM] = []
        for vm_id in range(cfg["vm_count"]):
            vm_cache = make_cache(cfg["vm_cache_size"], cfg["vm_cache_policy"])
            vm = VM(vm_id, vm_cache, cfg["vm_latency"], self.hypervisor)
            self.vms.append(vm)

        self.workload = self._build_workload(cfg["workload"])
        self.access_log = []

    def _build_workload(self, wcfg: Dict) -> List[Dict]:
        mode = wcfg.get("mode", "random")
        if mode == "provided":
            return wcfg.get("accesses", [])
        elif mode == "random":
            rnd = wcfg.get("random", {})
            length = rnd.get("length", 100)
            files = rnd.get("files", ["A","B","C","D","E","F","G"])
            res = []
            for _ in range(length):
                vm = random.randrange(self.cfg["vm_count"])
                file_id = random.choice(files)
                res.append({"vm": vm, "file": file_id})
            return res
        raise ValueError("Unknown workload mode")

    def run(self):
        for step, op in enumerate(self.workload):
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
                "host_cache_size": self.host_cache.capacity
            })

    def collect_results(self) -> Dict:
        vms_stats = [vm.stats() for vm in self.vms]
        host_info = self.hypervisor.host_cache.info()
        host_metrics = {
            "host_hits": self.hypervisor.host_hits,
            "disk_fetches": self.hypervisor.disk_fetches,
            "host_cache_info": host_info
        }
        totals = {
            "total_accesses": len(self.workload),
            "total_latency": sum(a["latency"] for a in self.access_log)
        }
        return {"vms": vms_stats, "host": host_metrics, "totals": totals, "access_log": self.access_log}

    def save_outputs(self, json_path: Optional[str], csv_path: Optional[str]):
        results = self.collect_results()
        if json_path:
            with open(json_path, "w") as f: json.dump(results, f, indent=4)
        if csv_path:
            with open(csv_path, "w", newline="") as f:
                fieldnames = ["step","vm","file","where","latency","vm_cache_size","host_cache_size"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in results["access_log"]: writer.writerow(row)

def load_config(path: str) -> Dict:
    with open(path, "r") as f: return json.load(f)

def main():
    if len(sys.argv) < 2:
        print("Uso: python3 cache_virtual.py config.json")
        sys.exit(1)
    cfg = load_config(sys.argv[1])
    sim = Simulator(cfg)
    sim.run()
    sim.save_outputs(cfg.get("output_json","results.json"), cfg.get("output_csv","results.csv"))

if __name__ == "__main__":
    main()