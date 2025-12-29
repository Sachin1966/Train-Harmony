import sqlite3
import os
from werkzeug.security import generate_password_hash, check_password_hash

DB_NAME = "railflow.db"

def get_db_connection():
    conn = sqlite3.connect(os.path.join(os.path.dirname(__file__), DB_NAME))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    
    # Create Users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL,
            status TEXT DEFAULT 'Offline',
            shift TEXT,
            location TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create Settings table
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_settings (
            user_id TEXT PRIMARY KEY,
            dark_mode BOOLEAN DEFAULT 1,
            compact_mode BOOLEAN DEFAULT 0,
            email_alerts BOOLEAN DEFAULT 1,
            push_notifications BOOLEAN DEFAULT 1,
            sound_effects BOOLEAN DEFAULT 0,
            language TEXT DEFAULT 'en',
            timezone TEXT DEFAULT 'UTC',
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    # Create Messages table
    c.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender_id TEXT,
            recipient_id TEXT,
            recipient_email TEXT,
            subject TEXT,
            body TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_read BOOLEAN DEFAULT 0,
            FOREIGN KEY (sender_id) REFERENCES users (id),
            FOREIGN KEY (recipient_email) REFERENCES users (email)
        )
    ''')
    
    # Check if admin exists, if not create default admin
    user = c.execute('SELECT * FROM users WHERE email = ?', ('admin@railflow.net',)).fetchone()
    if not user:
        # Create default admin
        # ID format: AD-001 for admin, OP-XXX for operators
        admin_id = 'AD-001'
        password_hash = generate_password_hash('admin123')
        c.execute('''
            INSERT INTO users (id, name, email, password, role, status, shift, location)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (admin_id, 'System Administrator', 'admin@railflow.net', password_hash, 'Admin', 'Active', 'All', 'HQ'))
        
        # Create default settings for admin
        c.execute('''
            INSERT INTO user_settings (user_id) VALUES (?)
        ''', (admin_id,))
        
        print("Default admin created: admin@railflow.net / admin123")
    
    conn.commit()
    conn.close()

def get_user_by_email(email):
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
    conn.close()
    return user

def get_user_by_id(user_id):
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    return user

def create_user(user_data):
    conn = get_db_connection()
    try:
        password_hash = generate_password_hash(user_data['password'])
        conn.execute('''
            INSERT INTO users (id, name, email, password, role, status, shift, location)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_data['id'],
            user_data['name'],
            user_data['email'],
            password_hash,
            user_data['role'],
            user_data.get('status', 'Offline'),
            user_data.get('shift', ''),
            user_data.get('location', '')
        ))
        # Initialize settings for new user
        conn.execute('INSERT INTO user_settings (user_id) VALUES (?)', (user_data['id'],))
        
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def get_all_operators():
    conn = get_db_connection()
    operators = conn.execute("SELECT id, name, email, role, status, shift, location FROM users WHERE role != 'Admin'").fetchall()
    conn.close()
    return [dict(row) for row in operators]

def get_user_settings(user_id):
    conn = get_db_connection()
    settings = conn.execute('SELECT * FROM user_settings WHERE user_id = ?', (user_id,)).fetchone()
    
    # If no settings found (e.g. old user), create default
    if not settings:
        conn.execute('INSERT INTO user_settings (user_id) VALUES (?)', (user_id,))
        conn.commit()
        settings = conn.execute('SELECT * FROM user_settings WHERE user_id = ?', (user_id,)).fetchone()
        
    conn.close()
    return dict(settings)

def update_user_settings(user_id, data):
    conn = get_db_connection()
    try:
        # Build update query dynamically based on provided keys
        valid_keys = ['dark_mode', 'compact_mode', 'email_alerts', 'push_notifications', 'sound_effects', 'language', 'timezone']
        updates = []
        values = []
        
        for key in valid_keys:
            if key in data:
                updates.append(f"{key} = ?")
                values.append(data[key])
                
        if not updates:
            return False
            
        values.append(user_id)
        query = f"UPDATE user_settings SET {', '.join(updates)} WHERE user_id = ?"
        
        conn.execute(query, values)
        conn.commit()
        return True
    except Exception as e:
        print(f"Error updating settings: {e}")
        return False
    finally:
        conn.close()

def update_user_profile(user_id, name, email):
    conn = get_db_connection()
    try:
        conn.execute('UPDATE users SET name = ?, email = ? WHERE id = ?', (name, email, user_id))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False # Email likely taken
    finally:
        conn.close()

def change_user_password(user_id, new_password):
    conn = get_db_connection()
    try:
        password_hash = generate_password_hash(new_password)
        conn.execute('UPDATE users SET password = ? WHERE id = ?', (password_hash, user_id))
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()

def create_message(sender_id, recipient_email, subject, body):
    conn = get_db_connection()
    try:
        # Try to find recipient ID from email if possible, though strict PK is ID
        recipient_data = conn.execute('SELECT id FROM users WHERE email = ?', (recipient_email,)).fetchone()
        recipient_id = recipient_data['id'] if recipient_data else None

        conn.execute('''
            INSERT INTO messages (sender_id, recipient_id, recipient_email, subject, body)
            VALUES (?, ?, ?, ?, ?)
        ''', (sender_id, recipient_id, recipient_email, subject, body))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error creating message: {e}")
        return False
    finally:
        conn.close()

def get_user_messages(user_email):
    conn = get_db_connection()
    # Fetch messages where this user is the recipient (by email)
    messages = conn.execute('''
        SELECT m.*, u.name as sender_name 
        FROM messages m
        LEFT JOIN users u ON m.sender_id = u.id
        WHERE m.recipient_email = ? 
        ORDER BY m.timestamp DESC
    ''', (user_email,)).fetchall()
    conn.close()
    return [dict(row) for row in messages]

def mark_message_read(message_id):
    conn = get_db_connection()
    conn.execute('UPDATE messages SET is_read = 1 WHERE id = ?', (message_id,))
    conn.commit()
    conn.close()

def update_user_status(user_id, status):
    conn = get_db_connection()
    try:
        conn.execute('UPDATE users SET status = ? WHERE id = ?', (status, user_id))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error updating status: {e}")
        return False
    finally:
        conn.close()
