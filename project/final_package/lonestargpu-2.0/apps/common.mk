BIN    		:= $(TOPLEVEL)/bin
INPUTS 		:= $(TOPLEVEL)/inputs

NVCC 		:= nvcc
GCC  		:= g++
CC := $(GCC)
#CUB_DIR := $(TOPLEVEL)/cub-1.1.1
CUB_DIR := $(TOPLEVEL)/cub-1.7.4

#COMPUTECAPABILITY := sm_20
COMPUTECAPABILITY := sm_50
ifdef debug
FLAGS := -arch=$(COMPUTECAPABILITY) -g -DLSGDEBUG=1 -G -maxrregcount=40 -D_FORCE_INLINES -rdc=true -lcudadevrt -lcudart -L /usr/lib/x86_64-linux-gnu
else
# including -lineinfo -G causes launches to fail because of lack of resources, pity.
FLAGS := -O3 -arch=$(COMPUTECAPABILITY) -Xptxas -v  -G -maxrregcount=40 -D_FORCE_INLINES -rdc=true -lcudadevrt -L /usr/lib/x86_64-linux-gnu #-lineinfo -G
endif
INCLUDES := -I $(TOPLEVEL)/include -I $(CUB_DIR)
LINKS := 

EXTRA := $(FLAGS) $(INCLUDES) $(LINKS)

.PHONY: clean variants support optional-variants

ifdef APP
$(APP): $(SRC) $(INC)
	$(NVCC) $(EXTRA) -DVARIANT=0 -o $@ $<
	cp $@ $(BIN)

variants: $(VARIANTS)

optional-variants: $(OPTIONAL_VARIANTS)

support: $(SUPPORT)

clean: 
	rm -f $(APP) $(BIN)/$(APP)
ifdef VARIANTS
	rm -f $(VARIANTS)
endif
ifdef OPTIONAL_VARIANTS
	rm -f $(OPTIONAL_VARIANTS)
endif

endif
