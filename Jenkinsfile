pipeline {
  agent { docker { image 'python:3.8.3' } }
  stages {
    stage('build') {
      steps {
        sh 'pip3 install -r requirements.txt'
      }
    }
    stage('test') {
      steps {
        sh 'python RestApiUnitTests.py'
      }   
    }
  }
}
