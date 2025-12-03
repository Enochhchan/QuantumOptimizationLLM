pipeline {
    agent any

    options {
        timestamps()
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Set up Python') {
            steps {
                bat '''
                py -3 -m pip install --upgrade pip
                py -3 -m pip install pytest
                if exist requirements.txt py -3 -m pip install -r requirements.txt
                '''
            }
        }

        stage('Run tests') {
            steps {
                bat '''
                py -3 -m pytest
                '''
            }
        }

        stage('Finish') {
            steps {
                echo 'Build and tests completed successfully on Jenkins.'
            }
        }
    }
}
