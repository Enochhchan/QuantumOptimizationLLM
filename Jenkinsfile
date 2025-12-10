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

            REM Try to build the manual with latexmk
            "C:\\Users\\enoch\\AppData\\Local\\Programs\\MiKTeX\\miktex\\bin\\x64\\latexmk.exe" -pdf -interaction=nonstopmode dsnManual.tex

            REM If latexmk failed (MiKTeX fresh install / permissions), warn but DO NOT fail pipeline
            IF ERRORLEVEL 1 (
            echo [WARN] LaTeX build failed in Jenkins (likely MiKTeX setup / permissions).
            echo [WARN] PDF is still built via GitHub Actions CI. Continuing pipeline...
            exit /B 0
            )
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
    }

    post {
        always {
            echo '[Post] Archiving artifacts'
            archiveArtifacts artifacts: 'latex/dsnManual.pdf, dist/**/*.zip', fingerprint: true
        }
        success {
            echo '[Post] Build succeeded!'
        }
        failure {
            echo '[Post] Build failed!'
        }
    }
}
