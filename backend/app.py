import os
import psycopg2 # type: ignore
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Função para conectar ao banco de dados PostgreSQL
def get_db_connection():
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=os.getenv('DB_PORT', '5432'),
        database=os.getenv('DB_NAME', 'Incidente'),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD', '1234')
    )
    conn.set_client_encoding('UTF8')  # Define o encoding da conexão como UTF-8
    return conn

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/')
def index():
    return jsonify({"message": "Welcome to the Incident Dashboard API"}), 200

# Endpoint para listar todos os incidentes
@app.route('/incidents', methods=['GET'])
def get_incidents():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM incidents;")
    incidents = cur.fetchall()
    cur.close()
    conn.close()

    # Convertendo para um formato de JSON
    incident_list = [
        {"id": row[0], "descricao": row[1], "categoria": row[2], "prioridade": row[3], "data_criacao": row[4]}
        for row in incidents
    ]
    return jsonify(incident_list)

# Endpoint para adicionar um novo incidente
@app.route('/incidents', methods=['POST'])
def add_incident():
    try:
        data = request.get_json()
        descricao = data.get('descricao')
        categoria = data.get('categoria', 'geral')
        prioridade = data.get('prioridade', 'baixa')

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO incidents (descricao, categoria, prioridade) VALUES (%s, %s, %s) RETURNING id;",
                    (descricao, categoria, prioridade)
                )
                incident_id = cur.fetchone()[0]
                conn.commit()
        
        return jsonify({"id": incident_id, "message": "Incident added successfully"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
