import pandas as pd

# Exemplo de dados fictícios de incidentes
data = {
    'descricao': [
        'Falha de rede na área central',
        'Erro de software na estação',
        'Problema de segurança detectado',
        'Servidor caiu',
        'Problema com infraestrutura física',
        'Sistema indisponível para os usuários',
        'Rede lenta em toda a empresa',
        'Atualização de segurança pendente',
        'Falta de energia no setor de TI',
        'Falha no backup do servidor principal'
    ],
    'categoria': ['rede', 'software', 'seguranca', 'infraestrutura', 'infraestrutura', 
                  'software', 'rede', 'seguranca', 'infraestrutura', 'infraestrutura'],
    'prioridade': ['alta', 'media', 'alta', 'alta', 'media', 'media', 'alta', 'baixa', 'media', 'alta']
}

# Converte para DataFrame
df = pd.DataFrame(data)
df.to_csv('dataset_incidentes.csv', index=False)
