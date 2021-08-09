class Hdf5Parallel < Formula
  desc "File format designed to store large amounts of data"
  homepage "https://www.hdfgroup.org/HDF5"
  url "https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.12/hdf5-1.12.1/src/hdf5-1.12.1.tar.bz2"
  sha256 "aaf9f532b3eda83d3d3adc9f8b40a9b763152218fa45349c3bc77502ca1f8f1c"

  keg_only "it conflicts with hdf5 package"

  depends_on "gcc"
  depends_on "mpich"
  depends_on "szip"
  
  fails_with :clang
  env :std
  
  def install
    ENV["CXX"] = "mpicxx"
    ENV["CC"] = "mpicc"
    
    system "env"
    system "./configure", "--enable-parallel",
                          "--disable-fortran",
                          "--prefix=#{prefix}"
    system "make", "install"    
  end
  
end
