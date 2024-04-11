
# FISHPACK source directory
FISHPACKDIR = ../../../external/fishpack4.1

# Compiler directive
COMPILE = gfortran -c -fPIC -std=legacy -freal-4-real-8 -fdefault-real-8

# F2PY directive
COMPILEPYF = f2py -c --f77flags="-freal-4-real-8 -fdefault-real-8 -std=legacy" --f90flags="-freal-4-real-8 -fdefault-real-8 -std=legacy" 

all:
	$(COMPILE) $(FISHPACKDIR)/src/comf.f -o comf.o
	$(COMPILEPYF) blktri.pyf $(FISHPACKDIR)/src/blktri.f -I. *.o
	
clean:
	rm -f *.o; rm -f *.so; rm -f *.pyc
