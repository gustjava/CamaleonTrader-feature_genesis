#!/usr/bin/env python3
"""
Script para verificar o status atual do estudo Optuna.
"""

import os
import sys
import sqlite3
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_optuna_status():
    """Verifica o status do estudo Optuna."""
    print("🔍 Verificando status do estudo Optuna...")
    
    # Caminho para o banco de dados do estudo
    study_db_path = project_root / "output" / "optuna_stage" / "study_a.sqlite"
    
    if not study_db_path.exists():
        print("❌ Banco de dados do estudo não encontrado:")
        print(f"   {study_db_path}")
        print("💡 Execute o pipeline primeiro para criar o estudo.")
        return
    
    try:
        # Conectar ao banco SQLite
        conn = sqlite3.connect(study_db_path)
        cursor = conn.cursor()
        
        # Verificar estudos disponíveis
        cursor.execute("SELECT study_id, study_name FROM studies")
        studies = cursor.fetchall()
        
        if not studies:
            print("❌ Nenhum estudo encontrado no banco de dados.")
            return
        
        print(f"📊 Estudos encontrados: {len(studies)}")
        for study_id, study_name in studies:
            print(f"   - ID: {study_id}, Nome: {study_name}")
        
        # Para cada estudo, verificar trials
        for study_id, study_name in studies:
            print(f"\n🔍 Analisando estudo: {study_name} (ID: {study_id})")
            
            # Contar trials por estado
            cursor.execute("""
                SELECT state, COUNT(*) 
                FROM trials 
                WHERE study_id = ? 
                GROUP BY state
            """, (study_id,))
            
            state_counts = cursor.fetchall()
            total_trials = 0
            
            print("📈 Status dos trials:")
            for state, count in state_counts:
                total_trials += count
                state_name = {
                    0: "RUNNING",
                    1: "COMPLETE", 
                    2: "PRUNED",
                    3: "FAIL"
                }.get(state, f"UNKNOWN({state})")
                print(f"   - {state_name}: {count}")
            
            print(f"📊 Total de trials: {total_trials}")
            
            # Verificar trials completos com valores
            cursor.execute("""
                SELECT trial_id, number, value, datetime_start, datetime_complete
                FROM trials 
                WHERE study_id = ? AND state = 1
                ORDER BY number DESC
                LIMIT 5
            """, (study_id,))
            
            recent_trials = cursor.fetchall()
            if recent_trials:
                print("🏆 Últimos 5 trials completos:")
                for trial_id, number, value, start, complete in recent_trials:
                    print(f"   - Trial {number}: valor={value:.6f}, completado em {complete}")
            
            # Verificar melhor trial
            cursor.execute("""
                SELECT trial_id, number, value
                FROM trials 
                WHERE study_id = ? AND state = 1
                ORDER BY value DESC
                LIMIT 1
            """, (study_id,))
            
            best_trial = cursor.fetchone()
            if best_trial:
                trial_id, number, value = best_trial
                print(f"🥇 Melhor trial: {number} com valor {value:.6f}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Erro ao verificar status: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_optuna_status()
