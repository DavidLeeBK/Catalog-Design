import os
import sqlite3
import json

DB_PATH = "data/simulator.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS simulation_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sim_name TEXT DEFAULT 'Untitled Simulation',
        run_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        num_firms INTEGER,
        num_plans INTEGER,
        welfare REAL,
        iterations INTEGER,
        results TEXT
    )
    """)
    conn.commit()
    conn.close()

def save_simulation(sim_name, num_firms, num_plans, welfare, iterations, results):
    """Insert a simulation run into the database."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO simulation_results (sim_name, num_firms, num_plans, welfare, iterations, results)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (sim_name, num_firms, num_plans, welfare, iterations, json.dumps(results)))
    conn.commit()
    conn.close()

def load_simulations():
    conn = sqlite3.connect(DB_PATH)
    df = None
    try:
        import pandas as pd
        df = pd.read_sql_query("SELECT * FROM simulation_results ORDER BY run_time DESC", conn)
    finally:
        conn.close()
    return df

# ============================================================
# Utilities for deletion
# ============================================================
def delete_simulation(sim_id: int):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM simulation_results WHERE id = ?", (sim_id,))
    conn.commit()
    conn.close()

def clear_database():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM simulation_results")
    conn.commit()
    conn.close()
