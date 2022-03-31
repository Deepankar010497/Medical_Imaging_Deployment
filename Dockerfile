FROM ubuntu:18.04
RUN apt upgrade && apt update && apt install -y \
    software-properties-common \
    python3-pip \
    python3-dev  
RUN ln -s /usr/bin/python3 /usr/bin/python
ADD . project/
RUN pip3 install --upgrade pip 
RUN cd project && pip3 install  --no-cache -r requirements.txt
EXPOSE 5000
ENTRYPOINT ["sh", "-c" ]
CMD ["cd project/ && chmod +x run.sh && ./run.sh"]