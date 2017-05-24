
# If nothing selected, use default config (this should set PLATFORM)
ifeq ($(CONFIG)$(PLATFORM),)
CONFIG=config.make
endif

ifndef PLATFORM
$(info Please select your target platform by running one of the following commands:)
$(info )
$(foreach mf, $(wildcard Makefile.*), $(info $(MAKE) PLATFORM=$(mf:Makefile.%=%)))
$(info )
$(error No PLATFORM given.)
else
include $(SRCDIR)Makefile.$(PLATFORM)
endif
