"""
数据中转站 — 本地存储 + 匿名化 + 云端同步（未来）
"""
import json
import time
import sqlite3
import hashlib
import os
from pathlib import Path
from collections import deque


class DataHub:
    """
    数据管理中心
    - 本地 SQLite 存储评分历史
    - 特征时序缓存
    - 匿名化数据包
    - 待实现: 联邦学习梯度上传
    """
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.expanduser("~/.hermes/skills_data.db")
        
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._feature_buffers = {}  # feature_name → deque
        self._init_db()
    
    def _init_db(self):
        """初始化数据库表"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    skill_name TEXT NOT NULL,
                    started_at REAL NOT NULL,
                    ended_at REAL,
                    duration_seconds REAL,
                    avg_score REAL,
                    best_score REAL,
                    metrics_json TEXT,
                    exported INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS anonymized_packets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    skill_name TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    features_json TEXT,
                    uploaded INTEGER DEFAULT 0,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            """)
            conn.commit()
    
    # ---- Session 管理 ----
    def start_session(self, skill_name: str) -> str:
        """开始新训练会话"""
        sid = hashlib.sha256(
            f"{skill_name}_{time.time()}_{os.urandom(4).hex()}".encode()
        ).hexdigest()[:16]
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO sessions (id, skill_name, started_at) VALUES (?, ?, ?)",
                (sid, skill_name, time.time())
            )
            conn.commit()
        
        self._feature_buffers = {}
        return sid
    
    def end_session(self, session_id: str, metrics: dict = None):
        """结束会话"""
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT started_at FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if row:
                duration = now - row[0]
                conn.execute(
                    """UPDATE sessions 
                       SET ended_at = ?, duration_seconds = ?, metrics_json = ?
                       WHERE id = ?""",
                    (now, duration, json.dumps(metrics or {}), session_id)
                )
                conn.commit()
    
    def save_metrics(self, session_id: str, metrics: dict):
        """实时保存当前指标"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE sessions SET metrics_json = ? WHERE id = ?",
                (json.dumps(metrics), session_id)
            )
            conn.commit()
    
    # ---- 特征缓存（内存中）----
    def add_features(self, features: dict):
        """添加一帧的特征数据到内存缓存"""
        for name, value in features.items():
            if name not in self._feature_buffers:
                self._feature_buffers[name] = deque(maxlen=600)  # 20s @ 30fps
            self._feature_buffers[name].append(value)
    
    def get_feature_history(self) -> dict:
        """获取所有特征的时序历史"""
        return {k: list(v) for k, v in self._feature_buffers.items()}
    
    # ---- 匿名化数据包 ----
    def create_anonymized_packet(self, session_id: str, skill_name: str,
                                  features: dict, metrics: dict) -> dict:
        """创建可用于云端上传的匿名数据包"""
        return {
            "skill": skill_name,
            "framework_version": "0.1.0",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "session_id": hashlib.sha256(session_id.encode()).hexdigest()[:12],
            "features": {
                k: list(v)[-50:] if hasattr(v, '__iter__') else v
                for k, v in features.items()
            },
            "metrics": {k: v.get("value") if isinstance(v, dict) else v
                        for k, v in (metrics or {}).items()},
            "metadata": {
                "duration_seconds": None,
                "fps": 30,
            }
        }
    
    def save_anonymized_packet(self, packet: dict):
        """保存匿名数据包到本地数据库"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO anonymized_packets 
                   (session_id, skill_name, timestamp, features_json)
                   VALUES (?, ?, ?, ?)""",
                (packet["session_id"], packet["skill"],
                 time.time(), json.dumps(packet["features"]))
            )
            conn.commit()
    
    # ---- 统计 ----
    def get_stats(self, skill_name: str = None) -> dict:
        """获取训练统计"""
        with sqlite3.connect(self.db_path) as conn:
            if skill_name:
                rows = conn.execute(
                    "SELECT COUNT(*), AVG(avg_score), MAX(best_score), SUM(duration_seconds) "
                    "FROM sessions WHERE skill_name = ? AND ended_at IS NOT NULL",
                    (skill_name,)
                ).fetchone()
            else:
                rows = conn.execute(
                    "SELECT COUNT(*), AVG(avg_score), MAX(best_score), SUM(duration_seconds) "
                    "FROM sessions WHERE ended_at IS NOT NULL"
                ).fetchone()
            
            return {
                "total_sessions": rows[0] or 0,
                "avg_score": round(rows[1] or 0, 1),
                "best_score": rows[2] or 0,
                "total_duration": round((rows[3] or 0) / 60, 1),
            }
