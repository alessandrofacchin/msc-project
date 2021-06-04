import numpy as np
import pytest

from latentneural.data import DataManager


def test_train_validation_test_split():
    data = np.random.randn(100,50,10,20)

    train, validation, test = DataManager.split_dataset(
        data,
        train_pct=0.7,
        val_pct=0.1,
        test_pct=0.2)
    assert train.shape == tuple([70] + list(data.shape[1:]))
    assert validation.shape == tuple([10] + list(data.shape[1:]))
    assert test.shape == tuple([20] + list(data.shape[1:]))

    train, validation, test = DataManager.split_dataset(
        data,
        train_pct=70,
        val_pct=10,
        test_pct=20)
    assert train.shape == tuple([70] + list(data.shape[1:]))
    assert validation.shape == tuple([10] + list(data.shape[1:]))
    assert test.shape == tuple([20] + list(data.shape[1:]))

    train, validation, test = DataManager.split_dataset(
        data,
        train_pct=0.7,
        val_pct=0.1,
        test_pct=0.2)
    assert train.shape == tuple([70] + list(data.shape[1:]))
    assert validation.shape == tuple([10] + list(data.shape[1:]))
    assert test.shape == tuple([20] + list(data.shape[1:]))

    train, validation, test = DataManager.split_dataset(
        data,
        train_pct=0.7,
        val_pct=0.1)
    assert train.shape == tuple([70] + list(data.shape[1:]))
    assert validation.shape == tuple([10] + list(data.shape[1:]))
    assert test.shape == tuple([20] + list(data.shape[1:]))

    train, validation, test = DataManager.split_dataset(
        data,
        train_pct=70,
        val_pct=10)
    assert train.shape == tuple([70] + list(data.shape[1:]))
    assert validation.shape == tuple([10] + list(data.shape[1:]))
    assert test.shape == tuple([20] + list(data.shape[1:]))

def test_train_validation_test_split_failures():
    data = np.random.randn(100,50,10,20)
    
    with pytest.raises(ValueError):
        DataManager.split_dataset(
        data,
        train_pct=70)

    with pytest.raises(ValueError):
        DataManager.split_dataset(
        data,
        train_pct=0.7)