FROM ubuntu:18.04
RUN apt upgrade && apt update && apt install -y \
    software-properties-common \
    python3-pip \
    python3-dev  
RUN ln -s /usr/bin/python3 /usr/bin/python
ADD . DEEP-LEARNING-PROJECT/
RUN pip3 install --upgrade pip 
RUN cd DEEP-LEARNING-PROJECT && pip3 install  --no-cache -r requirements.txt
EXPOSE 5000
ENTRYPOINT ["sh", "-c" ]
CMD ["cd DEEP-LEARNING-PROJECT/ && chmod +x run.sh && ./run.sh"]