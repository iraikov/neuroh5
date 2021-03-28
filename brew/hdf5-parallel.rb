class Hdf5Parallel < Formula
  desc "File format designed to store large amounts of data"
  homepage "https://www.hdfgroup.org/HDF5"
#   head "https://bitbucket.hdfgroup.org/scm/hdffv/hdf5.git"
  url "https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.12/hdf5-1.12.0/src/hdf5-1.12.0.tar.bz2"
  sha256 "97906268640a6e9ce0cde703d5a71c9ac3092eded729591279bf2e3ca9765f61"

  keg_only "it conflicts with hdf5 package"

  depends_on "gcc"
  depends_on "open-mpi"
  depends_on "szip"
  
  fails_with :clang
  env :std
  
  def install
    ENV["OMPI_CXX"] = ENV["CXX"]
    ENV["CXX"] = "mpicxx"
    ENV["OMPI_CC"] = ENV["CC"]
    ENV["CC"] = "mpicc"
    
    system "env"
    system "./configure", "--enable-parallel",
                          "--disable-fortran",
                          "--prefix=#{prefix}"
    system "make", "install"    
  end
  
end
