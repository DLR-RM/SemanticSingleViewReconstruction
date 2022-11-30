

unsigned char filterClassId(unsigned char value){
	switch(value){
		case 0: case 5: case 7: case 11: case 15: case 26:
		case 29: case 43: case 45: case 54: case 59: case 61:
		case 64: case 66: case 73: case 77: case 85: case 90:
		case 92: case 97:
		return 0; // is void 

		case 3: case 12: case 36: case 68: case 72: case 80:
		case 81:
		return 1; // is table 

		case 4: case 13: case 21: case 38: case 39: case 40:
		case 44: case 49: case 57: case 58: case 60: case 63:
		case 75: case 82: case 93: case 95: case 98:
		return 2; // is wall 

		case 42:
		return 3; // is bath 

		case 9: case 10: case 16: case 18: case 46: case 89:
		case 96:
		return 4; // is sofa 

		case 2: case 6: case 22: case 23: case 24: case 28:
		case 41: case 47: case 48: case 55: case 56: case 65:
		case 69: case 74: case 76: case 84: case 91:
		return 5; // is cabinet 

		case 17: case 20: case 67: case 70: case 87: case 88:
		return 6; // is bed 

		case 14: case 25: case 35: case 50: case 71: case 78:
		case 83: case 94:
		return 7; // is chair 

		case 1: case 27: case 30: case 31: case 37: case 51:
		case 52: case 53: case 62: case 79:
		return 8; // is floor 

		case 8: case 19: case 32: case 33: case 34: case 86:
		return 9; // is lighting 

	}
	std::cout << "This value was not processed: " << (int) value << std::endl;
	exit(1);
}



unsigned char filterClassIdReplica(unsigned char value){
	switch(value){
		case 1: case 2: case 4: case 6: case 8: case 10:
		case 11: case 12: case 13: case 15: case 16: case 17:
		case 18: case 19: case 21: case 22: case 23: case 24:
		case 25: case 26: case 27: case 28: case 29: case 30:
		case 33: case 34: case 35: case 36: case 37: case 38:
		case 39: case 40: case 41: case 42: case 44: case 45:
		case 49: case 50: case 51: case 52: case 53: case 54:
		case 55: case 57: case 58: case 59: case 60: case 63:
		case 64: case 65: case 66: case 69: case 70: case 73:
		case 74: case 75: case 76: case 78: case 79: case 80:
		case 81: case 82: case 83: case 84: case 85: case 86:
		case 88: case 89: case 90: case 91: case 92: case 95:
		case 96: case 98: case 99: case 101: case 102: case 103:
		case 104: case 105:
		return 0; // is void 

		case 46: case 72: case 100:
		return 1; // is table 

		case 14: case 32: case 47: case 68:
		return 2; // is wall 

		case 56:
		return 3; // is bath 

		case 97: case 0:
		return 4; // is sofa 

		case 3: case 7: case 9: case 31: case 43: case 62:
		case 93:
		return 5; // is cabinet 

		case 94:
		return 6; // is bed 

		case 5: case 20: case 71:
		return 7; // is chair 

		case 48: case 61: case 67: case 87:
		return 8; // is floor 

		case 77:
		return 9; // is lighting 

	}
	std::cout << "This value was not processed: " << (int) value << std::endl;
	exit(1);
}

unsigned char amountOfClasses(){ return 10; }