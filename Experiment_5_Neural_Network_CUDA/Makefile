NVCC = nvcc
NVCCFLAGS = -O2

TARGET = neural_network

all: $(TARGET)

$(TARGET): src/neural_network.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGET) $(TARGET).exe $(TARGET).obj $(TARGET).lib $(TARGET).exp

.PHONY: all clean
