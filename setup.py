#!/usr/bin/env python

from distutils.core import setup


setup(name="model",
      version="0.0.0",
      description="Multilayer spectral clustering for CITEseq datasets",
      author="Jacob Leistico",
      url="",
      packages=["model", "model.cbmc", "model.bmnc"],
      )
