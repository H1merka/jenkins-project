pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Setup') {
            steps {
                // Создаём виртуальное окружение и ставим зависимости
                sh 'python3 -m venv venv'
                sh 'venv/bin/pip install --upgrade pip setuptools'
                sh 'venv/bin/pip install -r requirements.txt'
            }
        }

        stage('Preprocess') {
            steps {
                // Download dataset from Kaggle
                withCredentials([string(credentialsId: 'kaggle-json', variable: 'KAGGLE_JSON')]) {
                    sh '''#!/bin/bash
                    mkdir -p ~/.kaggle
                    echo "$KAGGLE_JSON" > ~/.kaggle/kaggle.json
                    chmod 600 ~/.kaggle/kaggle.json
                    
                    venv/bin/pip install --no-cache-dir kaggle
                    venv/bin/python scripts/download_kaggle.py
                    '''
                }
                // Локальный препроцессинг
                sh 'PYTHONPATH=. venv/bin/python scripts/preprocess.py'
            }
        }

        stage('Train') {
            steps {
                sh 'PYTHONPATH=. venv/bin/python scripts/train.py'
            }
            post {
                success {
                    archiveArtifacts artifacts: 'model_bundle.pkl, lr_pipeline.pkl, best_model.txt', fingerprint: true
                }
            }
        }

        stage('Build Docker image') {
            steps {
                // Собираем образ с релевантным именем (amazon-sales)
                sh "docker build -t ml-amazon-sales:${env.BUILD_NUMBER} ."
            }
        }

        stage('Run Service') {
            steps {
                // Запускаем контейнер с моделью на порту 8000, избегая конфликта с Nginx (порт 80)
                sh "docker rm -f ml-amazon-sales || true"
                sh "docker run -d --name ml-amazon-sales -p 8000:80 ml-amazon-sales:${env.BUILD_NUMBER}"
            }
        }

        stage('Smoke Test') {
            steps {
                // Даем контейнеру 3 секунды на инициализацию FastAPI
                sh 'sleep 3'
                
                // Явно указываем интерпретатор bash для корректной работы операторов [[ ]]
                sh '''#!/bin/bash
                RESPONSE=$(curl -sS -X POST http://127.0.0.1:8000/predict \
                -H "Content-Type: application/json" \
                -d '{"inputs":[{"price": 100.0, "discount_percent": 5.0, "quantity_sold": 2, "rating": 4.5, "review_count": 100, "product_category": 1, "customer_region": 1, "payment_method": 1, "order_year": 2023, "order_month": 5}]}')

                echo "Response from API: $RESPONSE"
                
                # Проверяем наличие критических ошибок или Nginx заглушек
                if [[ "$RESPONSE" == *"error"* ]] || [[ "$RESPONSE" == *"405 Not Allowed"* ]]; then
                    echo "Smoke test failed! API returned an error."
                    exit 1
                fi
                
                # Проверяем успешность предикта
                if [[ "$RESPONSE" == *"predictions"* ]]; then
                    echo "Smoke test PASSED! Model is working."
                else
                    echo "Unexpected response format!"
                    exit 1
                fi
                '''
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: 'best_model.txt', allowEmptyArchive: true
        }
    }
}
