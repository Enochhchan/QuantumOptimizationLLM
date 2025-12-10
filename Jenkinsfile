pipeline {
    agent any

    options {
        timestamps()
    }

    stages {
        stage('Checkout') {
            steps {
                // Pull code from GitHub
                checkout scm
            }
        }

        stage('Set up Python venv') {
            steps {
                bat '''
                echo [Set up Python venv]
                py -3 -m venv venv
                call venv\\Scripts\\activate.bat
                python -m pip install --upgrade pip
                python -m pip install -r requirements.txt
                '''
            }
        }

        stage('Run Tests') {
            steps {
                bat '''
                echo [Run pytest]
                call venv\\Scripts\\activate.bat
                pytest
                '''
            }
        }

        stage('Build LaTeX PDF') {
            steps {
                bat '''
                echo [Build LaTeX PDF]
                cd latex
                latexmk -pdf -interaction=nonstopmode dsnManual.tex
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                bat '''
                echo [Build Docker image for web UI]
                docker build -t nl-qubo-web:latest .
                '''
            }
        }

        stage('Package Versioned Artifact') {
            steps {
                bat '''
                echo [Package versioned ZIP artifact]
                setlocal EnableDelayedExpansion

                for /f %%v in (version.txt) do set VERSION=%%v
                echo Detected version: !VERSION!

                if not exist dist mkdir dist

                powershell -Command "Compress-Archive -Path 'latex\\\\dsnManual.pdf' -DestinationPath 'dist\\\\QuantumOptimizationLLM_v!VERSION!.zip' -Force"

                endlocal
                '''
            }
        }
