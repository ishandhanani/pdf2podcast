 docker build -t agent-service .


 docker run -p 8964:8964 -e NIM_KEY=nimkey  agent-service