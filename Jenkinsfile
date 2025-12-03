pipeline {
    agent any

    options {
        timestamps()
    }

    stages {
        stage('Checkout') {
            steps {
                // This pulls the same repo the Jenkinsfile is in
                checkout scm
            }
        }

        stage('Set up Python') {
            steps {
                // Try python3 first, fall back to python
                sh '''
                which python3 && python3 -m pip install --upgrade pip || true
                which pip3 && pip3 install pytest || true

                # If you have a requirements.txt, install it
                if [ -f requirements.txt ]; then
                  pip3 install -r requirements.txt || pip install -r requirements.txt
                fi
                '''
            }
        }

        stage('Run tests') {
            steps {
                sh '''
                pytest
                '''
            }
        }

        stage('Archive test results (optional)') {
            steps {
                // If you later add reports, you can archive them here
                echo 'Build and tests finished. Jenkins autobuild succeeded.'
            }
        }
    }
}
