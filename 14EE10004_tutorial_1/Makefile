NVCC = nvcc

helloCUDA: headers.h kernel.cu host.cu  
	$(NVCC) -I ./ host.cu kernel.cu -o helloCUDA

      
run:
	./helloCUDA

clean:
	rm -rf helloCUDA
