conda-update:
	--conda update conda --all

conda-create:
	--conda env create -f ./environment.yml

conda-update:
	--conda env update --name latentneural -f ./environment.yml