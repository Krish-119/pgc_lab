NVCC = nvcc
NVCCFLAGS = -O2

TARGET = exp6

all: $(TARGET)

$(TARGET): src/exp6.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGET) $(TARGET).exe $(TARGET).obj $(TARGET).lib $(TARGET).exp

.PHONY: all clean
