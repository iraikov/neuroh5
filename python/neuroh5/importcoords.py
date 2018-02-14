'''
This script reads a text file in SWC format and creates a
representation of it in an HDF5 file.
'''

from mpi4py import MPI
import h5py
import numpy as np
import sys, os
import click
import itertools
from neuroh5.io import write_cell_attributes, append_cell_attributes

grp_h5types   = 'H5Types'
ns_coords     = 'Coordinates'

attr_x_coord     = 'X Coordinate'
attr_y_coord     = 'Y Coordinate'
attr_z_coord     = 'Z Coordinate'

attr_u_coord     = 'U Coordinate'
attr_v_coord     = 'V Coordinate'
attr_l_coord     = 'L Coordinate'

comm = MPI.COMM_WORLD

def import_xyz_uvl_coords (header,offset,lines,colsep,population,outputfile,iosize):

    l_gid = []
    l_x_coord = []
    l_y_coord = []
    l_z_coord = []
    l_u_coord = []
    l_v_coord = []
    l_l_coord = []

    if header:
        lines1 = lines[1:]
    else:
        lines1 = lines

    lst = filter(None,lines1[0].split(colsep))
    # read and parse line-by-line
    if len(lst) > 6:
        for l in lines1:
            if not(l.startswith("#")):
                a = filter(None, l.split(colsep))
                try:
                    l_gid.append(int(float(a[0])))
                except ValueError:
                    l_gid.append(int(a[0]))
                l_x_coord.append(float(a[1]))
                l_y_coord.append(float(a[2]))
                l_z_coord.append(float(a[3]))
                l_u_coord.append(float(a[4]))
                l_v_coord.append(float(a[5]))
                l_l_coord.append(float(a[6]))
    else:
        for l in lines1:
            if not(l.startswith("#")):
                a = filter(None, l.split(colsep))
                try:
                    l_gid.append(int(float(a[0])))
                except ValueError:
                    l_gid.append(int(int(a[0])))
                l_x_coord.append(float(a[1]))
                l_y_coord.append(float(a[2]))
                l_z_coord.append(float(a[3]))
                l_u_coord.append(float(a[4]))
                l_v_coord.append(float(a[5]))

    values = {}

    if len(l_l_coord) > 0:
        for (gid,x,y,z,u,v,l) in itertools.izip(l_gid,l_x_coord,l_y_coord,l_z_coord,l_u_coord,l_v_coord,l_l_coord):
            gid1 = gid + offset
            values[gid1] = {attr_x_coord:np.asarray([x]).astype('float32'),
                            attr_y_coord:np.asarray([y]).astype('float32'),
                            attr_z_coord:np.asarray([z]).astype('float32'),
                            attr_u_coord:np.asarray([u]).astype('float32'),
                            attr_v_coord:np.asarray([v]).astype('float32'),
                            attr_l_coord:np.asarray([l]).astype('float32')}
    else:
        for (gid,x,y,z,u,v) in itertools.izip(l_gid,l_x_coord,l_y_coord,l_z_coord,l_u_coord,l_v_coord):
            gid1 = gid + offset
            values[gid1] = {attr_x_coord:np.asarray([x]).astype('float32'),
                            attr_y_coord:np.asarray([y]).astype('float32'),
                            attr_z_coord:np.asarray([z]).astype('float32'),
                            attr_u_coord:np.asarray([u]).astype('float32'),
                            attr_v_coord:np.asarray([v]).astype('float32')}
    
    append_cell_attributes(comm, outputfile, population, values, io_size=iosize, namespace=ns_coords)


@click.command()
@click.argument("population", type=str)
@click.argument("inputfiles", type=click.Path(exists=True), nargs=-1)
@click.argument("outputfile", type=click.Path())
@click.option("--header", is_flag=True)
@click.option("--colsep", type=str, default=' ')
@click.option("--offset", type=int, default=0)
@click.option("--iosize", type=int, default=1)
@click.option("--bufsize", type=int, default=1000000)
def cli(inputfiles, outputfile, population, header, colsep, offset, iosize, bufsize):

    skip_header = header
    startindex = offset
    for inputfile in inputfiles:

        click.echo("Importing file: %s\n" % inputfile) 
        f = open(inputfile)
        lines = f.readlines(bufsize)
        
        while lines:
            import_xyz_uvl_coords(skip_header, offset, lines, colsep, population, outputfile, iosize)
            lines = f.readlines(bufsize)
            skip_header = False

        f.close()

        

