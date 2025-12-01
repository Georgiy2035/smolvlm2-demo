# smolvlm2-demo
Repository provides smolvlm2 access using gradio interface, MIPT python course project.

# Instalation
run inside repo: 

docker-compose up --build

# Using

docker run -p 7860:7860 \
  -v ./models:/app/models \
  -e DEVICE=cpu \
  smolvlm2-demo

The server will be on 0.0.0.0:7860

Solved tasks: VQA + Captioning and OCR.
