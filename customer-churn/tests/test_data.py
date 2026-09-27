from src.data_processor import FEATURE_COLUMNS, TARGET_COLUMN, generate_data


def test_generated_dataset_has_expected_columns_and_no_nulls():
    dataset = generate_data()

    assert len(dataset) == 1000
    assert list(dataset.columns) == FEATURE_COLUMNS + [TARGET_COLUMN]
    assert dataset.notna().all().all()