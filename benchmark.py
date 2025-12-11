#!/usr/bin/env python3
"""
Script Benchmark e Visualização
Uso: 
    python3 benchmark_script.py config.json          (Roda apenas a política do JSON)
    python3 benchmark_script.py config.json --all    (Roda FIFO, LRU e LFU e compara)
"""

import sys
import json
import subprocess
import argparse
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. FUNÇÕES DE VISUALIZAÇÃO
# ==========================================

def plot_hit_rates(metrics: dict):
    """Gera gráfico de barras de Hit Rates"""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        policies = list(metrics.keys())
        x = np.arange(len(policies))
        width = 0.25
        
        vm_rates = [metrics[p]['vm_hit_rate'] for p in policies]
        host_rates = [metrics[p]['host_hit_rate'] for p in policies]
        disk_rates = [metrics[p]['disk_hit_rate'] for p in policies]
        
        ax.bar(x - width, vm_rates, width, label='VM Cache', color='#2ecc71')
        ax.bar(x, host_rates, width, label='Host Cache', color='#3498db')
        ax.bar(x + width, disk_rates, width, label='Disco', color='#e74c3c')
        
        ax.set_ylabel('Taxa de Acerto (%)', fontsize=12)
        ax.set_title('Hit Rates por Nível de Cache', fontsize=14, fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels(policies)
        ax.legend(); ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('graph_hit_rates.png', dpi=300)
        print("  [V] Gráfico salvo: graph_hit_rates.png")
        plt.close()
    except Exception as e:
        print(f"Erro ao gerar graph_hit_rates: {e}")

def plot_latency_comparison(metrics: dict):
    """Gera gráficos de latência"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        policies = list(metrics.keys())
        total_lat = [metrics[p]['total_latency'] for p in policies]
        avg_lat = [metrics[p]['avg_latency'] for p in policies]
        
        # Cores dinâmicas (se for 1 barra ou 3)
        colors = ['#e74c3c', '#3498db', '#2ecc71'] if len(policies) > 1 else ['#3498db']
        
        ax1.bar(policies, total_lat, color=colors[:len(policies)])
        ax1.set_title('Latência Total', fontweight='bold')
        for i, v in enumerate(total_lat): ax1.text(i, v, str(v), ha='center', va='bottom')
        
        ax2.bar(policies, avg_lat, color=colors[:len(policies)])
        ax2.set_title('Latência Média', fontweight='bold')
        for i, v in enumerate(avg_lat): ax2.text(i, v, f"{v:.2f}", ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('graph_latency.png', dpi=300)
        print("  [V] Gráfico salvo: graph_latency.png")
        plt.close()
    except Exception as e:
        print(f"Erro ao gerar graph_latency: {e}")

def plot_evictions(metrics: dict):
    """Gera gráfico de Evictions"""
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        policies = list(metrics.keys())
        evictions = [metrics[p]['total_evictions'] for p in policies]
        
        bars = ax.bar(policies, evictions, color='#f39c12')
        ax.set_title('Thrashing: Total de Evictions', fontweight='bold')
        ax.set_ylabel('Quantidade de Evicções')
        ax.bar_label(bars)
        plt.tight_layout()
        plt.savefig('graph_evictions.png', dpi=300)
        print("  [V] Gráfico salvo: graph_evictions.png")
        plt.close()
    except Exception as e:
        print(f"Erro ao gerar graph_evictions: {e}")

def plot_timeline(log: list, policy: str):
    """Gera timeline de acessos"""
    try:
        steps = [e['step'] for e in log]
        lats = [e['latency'] for e in log]
        colors = {'vm': '#2ecc71', 'host': '#3498db', 'disk': '#e74c3c'}
        c_map = [colors[e['where']] for e in log]
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.scatter(steps, lats, c=c_map, s=80, alpha=0.7, edgecolors='black')
        ax.set_title(f'Timeline de Acessos - {policy.upper()}', fontweight='bold')
        ax.set_xlabel('Passo (Sequência de Acesso)')
        ax.set_ylabel('Latência')
        
        # Legenda Manual
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#2ecc71', label='VM Hit'),
                           Patch(facecolor='#3498db', label='Host Hit'),
                           Patch(facecolor='#e74c3c', label='Disk Fetch')]
        ax.legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.savefig(f'graph_timeline_{policy.lower()}.png', dpi=300)
        print(f"  [V] Gráfico salvo: graph_timeline_{policy.lower()}.png")
        plt.close()
    except Exception as e:
        print(f"Erro ao gerar timeline de {policy}: {e}")

def plot_efficiency(metrics: dict):
    """Gráfico de eficiência (Stacked Bar)"""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        policies = list(metrics.keys())
        x = np.arange(len(policies))
        
        vm_rates = np.array([metrics[p]['vm_hit_rate'] for p in policies])
        host_rates = np.array([metrics[p]['host_hit_rate'] for p in policies])
        disk_rates = np.array([metrics[p]['disk_hit_rate'] for p in policies])
        
        ax.bar(x, vm_rates, label='VM Cache', color='#2ecc71')
        ax.bar(x, host_rates, bottom=vm_rates, label='Host Cache', color='#3498db')
        ax.bar(x, disk_rates, bottom=vm_rates+host_rates, label='Disco', color='#e74c3c')
        
        ax.set_ylabel('Distribuição (%)')
        ax.set_title('Eficiência de Cache', fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels(policies)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('graph_efficiency.png', dpi=300)
        print("  [V] Gráfico salvo: graph_efficiency.png")
        plt.close()
    except Exception as e:
        print(f"Erro ao gerar graph_efficiency: {e}")

# ==========================================
# 2. LÓGICA DE EXECUÇÃO
# ==========================================

def load_config(path: str) -> Dict:
    with open(path) as f: return json.load(f)

def run_simulation(config: Dict, policy: str, output_suffix: str) -> Dict:
    # Ajusta config para a política atual
    config['vm_cache_policy'] = policy
    config['host_cache_policy'] = policy
    config['output_json'] = f'results_{output_suffix}.json'
    config['output_csv'] = f'results_{output_suffix}.csv'
    
    # Cria arquivo temporário
    temp_config = f'config_temp_{output_suffix}.json'
    with open(temp_config, 'w') as f: json.dump(config, f, indent=4)
    
    # Chama o script cache_virtual.py
    subprocess.run([sys.executable, 'cache_virtual.py', temp_config], check=True)
    
    # Lê os resultados
    with open(config['output_json']) as f: results = json.load(f)
    Path(temp_config).unlink()
    return results

def calculate_metrics(results: Dict) -> Dict:
    vm = results['vms'][0]
    host = results['host']
    total = vm['accesses']
    return {
        'total_accesses': total,
        'vm_hits': vm['vm_hits'],
        'vm_hit_rate': (vm['vm_hits'] / total * 100) if total else 0,
        'host_hits': vm['host_hits'],
        'host_hit_rate': (vm['host_hits'] / total * 100) if total else 0,
        'disk_hits': vm['disk_hits'],
        'disk_hit_rate': (vm['disk_hits'] / total * 100) if total else 0,
        'total_evictions': vm['vm_cache_info']['evictions'] + host['host_cache_info']['evictions'],
        'avg_latency': results['totals']['total_latency'] / total if total else 0,
        'total_latency': results['totals']['total_latency']
    }

def print_clean_table(metrics: Dict[str, Dict]):
    print("\n" + "="*65)
    print("RESULTADOS DA SIMULAÇÃO")
    print("="*65)
    
    # Cabeçalho dinâmico
    policies = list(metrics.keys())
    headers = "".join([f"{p:>12}" for p in policies])
    print(f"{'Métrica':<20} {headers}")
    print("-"*65)
    
    row_keys = [
        ('VM Hit Rate (%)', 'vm_hit_rate', True),
        ('Host Hit Rate (%)', 'host_hit_rate', True),
        ('Disk Hit Rate (%)', 'disk_hit_rate', True),
        ('Total Evictions', 'total_evictions', False),
        ('Latência Total', 'total_latency', False),
        ('Latência Média', 'avg_latency', False)
    ]
    
    for label, key, is_pct in row_keys:
        row_str = f"{label:<20} "
        for p in policies:
            val = metrics[p][key]
            if is_pct:
                row_str += f"{val:>12.1f}"
            elif key == 'avg_latency':
                row_str += f"{val:>12.2f}"
            else:
                row_str += f"{val:>12}"
        print(row_str)
    print("="*65)

def main():
    # Configuração do Argparse
    parser = argparse.ArgumentParser(description="Simulador de Cache (Benchmark)")
    parser.add_argument("config", help="Arquivo de configuração JSON")
    parser.add_argument("--all", action="store_true", help="Executa benchmark comparativo (FIFO, LRU, LFU)")
    
    args = parser.parse_args()
    
    base_config = load_config(args.config)
    
    # Decide quais políticas rodar
    if args.all:
        policies = ['FIFO', 'LRU', 'LFU']
        print(f"Modo: Benchmark Comparativo ({', '.join(policies)})")
    else:
        # Pega a política definida no JSON (Padrão) ou assume FIFO se não existir
        default_policy = base_config.get("vm_cache_policy", "FIFO")
        policies = [default_policy]
        print(f"Modo: Execução Simples ({default_policy})")
    
    all_metrics = {}
    access_logs = {}
    
    # Loop de execução
    for policy in policies:
        print(f"-> Simulando {policy}...")
        # Se for modo simples, usa o nome da política; se for comparativo, usa ela mesma
        results = run_simulation(base_config.copy(), policy, policy.lower())
        all_metrics[policy] = calculate_metrics(results)
        access_logs[policy] = results['access_log']
    
    # 1. Imprime Tabela (Dinâmica para 1 ou 3 colunas)
    print_clean_table(all_metrics)
    
    # 2. Gera os Gráficos
    print("\nGerando gráficos...")
    # Gráficos de barra funcionam mesmo com 1 item, mas mostram só 1 barra
    plot_hit_rates(all_metrics)
    plot_latency_comparison(all_metrics)
    plot_evictions(all_metrics)
    plot_efficiency(all_metrics)
    
    # Gera timelines individuais
    for policy in policies:
        plot_timeline(access_logs[policy], policy)
        
    print("\nProcesso concluído! Verifique os arquivos .png gerados.")

if __name__ == "__main__":
    main()