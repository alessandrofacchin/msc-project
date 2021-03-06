import os
import runpy
import pytest

from latentneural.utils import upsert_empty_folder, logger, remove_folder


@pytest.fixture(scope='module')
def notebooks_converted():
    test_notebooks_folder = os.path.join('test', 'notebooks', 'deliverables')
    original_notebooks_folder = os.path.join('notebooks', 'deliverables')

    upsert_empty_folder(test_notebooks_folder)

    logger.info('Finding deliverables notebooks')
    notebooks = [f for f in os.listdir(original_notebooks_folder) if os.path.isfile(
        os.path.join(original_notebooks_folder, f)) and f[-6:] == '.ipynb']

    for notebook in notebooks:
        os.system('jupyter nbconvert --to python --output "%s" "%s"' % (
            os.path.join(
                '..',
                '..',
                test_notebooks_folder,
                notebook).replace(
                '.ipynb',
                '.py'),
            os.path.join(original_notebooks_folder, notebook),
        ))
        new_filename = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                'deliverables',
                notebook).replace(
                '.ipynb',
                '.py')
        with open(new_filename, 'r') as fp:
            lines = fp.readlines()
        lines = [x.replace('epochs=1000', 'epochs=1') for x in lines]
        lines = [x for x in lines if ".show(" not in x]
        with open(new_filename, 'w') as fp:
            fp.writelines(lines)
    return True


@pytest.fixture(scope='function')
def lorenz_generator_filename(notebooks_converted):
    if notebooks_converted:
        return os.path.join('.', 'test', 'notebooks',
                            'deliverables', 'lorenz_generator.py')

@pytest.fixture(scope='function')
def lfads_on_lorenz_filename(notebooks_converted):
    if notebooks_converted:
        return os.path.join('.', 'test', 'notebooks',
                            'deliverables', 'lfads_on_lorenz.py')

@pytest.fixture(scope='function')
def tndm_on_lorenz_filename(notebooks_converted):
    if notebooks_converted:
        return os.path.join('.', 'test', 'notebooks',
                            'deliverables', 'tndm_on_lorenz.py')

@pytest.fixture(scope='module', autouse=True)
def cleanup(request):
    def remove_test_dir():
        test_notebooks_folder = os.path.join(
            'test', 'notebooks', 'deliverables')
        remove_folder(test_notebooks_folder)
    request.addfinalizer(remove_test_dir)

@pytest.mark.notebook
@pytest.mark.smoke
def test_smoke_lorenz_generator(lorenz_generator_filename):
    runpy.run_path(lorenz_generator_filename)
    assert True

@pytest.mark.notebook
@pytest.mark.smoke
def test_smoke_lfads_on_lorenz(lfads_on_lorenz_filename):
    runpy.run_path(lfads_on_lorenz_filename)
    assert True

@pytest.mark.notebook
@pytest.mark.smoke
def test_smoke_tndm_on_lorenz(tndm_on_lorenz_filename):
    runpy.run_path(tndm_on_lorenz_filename)
    assert True