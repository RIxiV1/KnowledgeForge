import sqlite3
from config import DATABASE_PATH

def get_connection():
    return sqlite3.connect( DATABASE_PATH, check_same_thread=False)


def create_tables():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            answer TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    # Registry of indexed files so dedup survives restarts and each file's
    # vectors can be removed individually (chunks are tagged with file_hash).
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS indexed_files (
            file_hash TEXT PRIMARY KEY,
            filename TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    conn.commit()
    conn.close()


def save_file(file_hash, filename):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO indexed_files (file_hash, filename) VALUES (?, ?)",
        (file_hash, filename),
    )
    conn.commit()
    conn.close()


def get_indexed_files():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT file_hash, filename FROM indexed_files ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_indexed_hashes():
    return {file_hash for file_hash, _ in get_indexed_files()}


def delete_file(file_hash):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM indexed_files WHERE file_hash = ?", (file_hash,))
    conn.commit()
    conn.close()


def clear_indexed_files():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM indexed_files")
    conn.commit()
    conn.close()

def save_chat(question, answer):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(""" INSERT INTO chat_history ( question, answer ) VALUES (?, ?) """, (question, answer) )
    conn.commit()
    conn.close()
    
def delete_old_chats(days=30):
    """Delete chats older than X days"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        DELETE FROM chat_history 
        WHERE created_at < datetime('now', '-' || ? || ' days')
    """, (days,))
    conn.commit()
    conn.close()

def get_chat_history():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute( """ SELECT question, answer, created_at FROM chat_history ORDER BY id DESC """ )
    rows = cursor.fetchall()
    conn.close()
    return rows

def clear_history():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute( "DELETE FROM chat_history")
    conn.commit()
    conn.close()