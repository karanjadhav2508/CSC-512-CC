INCLUDE_DIR   =./include
SRC_DIR       =./src
OBJECTS_DIR   =./objects
BIN_DIR       =./bin

OBJS = ${OBJECTS_DIR}/first_plus.o \
       ${OBJECTS_DIR}/parser.o     \
       $(OBJECTS_DIR)/token.o      \
       $(OBJECTS_DIR)/scanner.o    \
       $(OBJECTS_DIR)/parse_main.o 

ifeq ($(DEBUG),1)
  DEBUG_FLAGS = -DDEBUG

else
  DEBUG_FLAGS =

endif


CC=/usr/bin/g++-4.8
CC_OPTS=-O3 -std=c++11 -I $(INCLUDE_DIR) $(DEBUG_FLAGS) -c

all : $(BIN_DIR) $(BIN_DIR)/parser

$(BIN_DIR)/parser : $(OBJS)
	$(CC) $^ -o $@

$(OBJECTS_DIR)/%.o : $(SRC_DIR)/%.cpp
	$(CC) $(CC_OPTS) $^ -o $@

$(BIN_DIR) :
	mkdir -p $@

clean :
	find . -name "*~" -print | xargs rm -f
	rm -f $(OBJECTS_DIR)/*
	rm -f $(BIN_DIR)/*

.PHONY : clean
