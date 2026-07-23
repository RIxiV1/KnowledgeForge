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