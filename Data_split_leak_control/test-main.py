import os
import pandas as pd
from data_splitter import DataSplitter, save_splits

def main():
    # Папка, где лежит скрипт
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Путь к файлу с датасетом
    dataset_path = os.path.join(current_dir, "data", "test-data.csv")

    df = pd.read_csv(dataset_path)

    # Создаём сплиттер
    splitter = DataSplitter()

    # Разделение данных и вывод отчёта
    train_df, val_df, test_df = splitter.split_dataset(df, target_column=None)

    # Сохраняем файлы
    save_splits(train_df, val_df, test_df, path=os.path.join(current_dir, "data"))


if __name__ == "__main__":
    main()