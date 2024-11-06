import pytest
from app import app, get_db_connection

@pytest.fixture
def client():
    # Configura o aplicativo para o modo de teste
    app.config['TESTING'] = True

    # Limpa o banco de dados antes de cada teste
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM incidents;")  # Apaga todos os registros da tabela
            conn.commit()

    # Cria o cliente de teste
    with app.test_client() as client:
        yield client
def test_add_incident(client):
    new_incident = {
        "descricao": "Incidente de teste",
        "categoria": "teste",
        "prioridade": "alta"
    }
    response = client.post('/incidents', json=new_incident)
    assert response.status_code == 201
    assert 'id' in response.json
    assert response.json['message'] == "Incident added successfully"

def test_get_incidents_with_data(client):
    # Primeiro, adiciona um incidente
    new_incident = {
        "descricao": "Incidente de teste",
        "categoria": "teste",
        "prioridade": "alta"
    }
    client.post('/incidents', json=new_incident)
    
    # Agora verifica se o incidente aparece no GET
    response = client.get('/incidents')
    assert response.status_code == 200
    assert len(response.json) > 0  # Espera-se que haja pelo menos um incidente
    assert response.json[0]['descricao'] == "Incidente de teste"
