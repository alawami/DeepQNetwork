docker run --gpus all --rm -it -v "$(pwd)":/rl -p 8787:8787 -p 8786:8786 -p 8888:8888 aalawami/banana_rl:cuda10.2 --
