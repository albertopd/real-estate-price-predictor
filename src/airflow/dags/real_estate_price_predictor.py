import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from datetime import timezone
from pathlib import Path
from typing import Any
from airflow import DAG
from airflow.decorators import task
from airflow.models import Variable
from ml.config.config import MLFlowConfig, ModelConfig
from ml.training.regression_trainer import RegressionTrainer
from ml.pipelines.analysis_preprocess import prepare_analysis_dataset
from ml.pipelines.training_preprocess import prepare_training_dataset
from scrapers.immovlan_listing_scraper import ImmovlanListingScraper
from scrapers.immovlan_sitemap_scraper import ImmovlanSitemapScraper
from utils.logging_utils import setup_logger


# Make project root importable for tasks
REPO_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[3]))
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SITEMAPS_DIR = RAW_DIR / "sitemaps"
ANALYSIS_DIR = DATA_DIR / "analysis"
TRAINING_DIR = DATA_DIR / "training"
MODELS_DIR = REPO_ROOT / "ml_models"


# DAG default arguments
default_args = {
    "owner": "data-eng", 
    "depends_on_past": False
}
#TODO: enable email alerts and retries
#    "retries": 1,
#    "retry_delay": timedelta(minutes=5),
#    "email_on_failure": True,
#    "email_on_retry": True,

with DAG(
    dag_id="real_estate_price_predictor",
    start_date=datetime(2025, 9, 7, tzinfo=timezone.utc),
    schedule="0 2 * * *",  # 02:00 every night
    catchup=False,
    default_args=default_args,
    description="Real estate price prediction pipeline",
    tags=[
        "real_estate",
        "scraping",
        "etl",
        "ml-training",
        "ml-evaluation",
        "nightly",
        "data-eng",
        "mlops",
    ],
    max_active_runs=1,
) as dag:

    @task
    def scrape_sitemaps() -> None:
        """Scrape and save sitemaps from immovlan.be."""
        logger = setup_logger(__name__)
        logger.info("Starting sitemap scraping...")

        try:
            SITEMAPS_DIR.mkdir(parents=True, exist_ok=True)
            scraper = ImmovlanSitemapScraper(SITEMAPS_DIR)
            scraper.scrape_sitemaps()
            logger.info("Sitemap scraping completed successfully.")

        except Exception as e:
            logger.error(f"Sitemap scraping failed: {str(e)}")
            raise

    @task
    def scrape_apartments() -> str:
        """Scrape apartment listings and save to parquet."""
        logger = setup_logger(__name__)
        logger.info("Starting apartment scraping...")

        try:
            urls_txt_file = SITEMAPS_DIR / "apartments_links.txt"
            if not urls_txt_file.exists():
                raise FileNotFoundError(
                    f"Apartment links file not found: {urls_txt_file}"
                )

            RAW_DIR.mkdir(parents=True, exist_ok=True)
            date_str = datetime.now(timezone.utc).date()
            output_file_path = RAW_DIR / "apartments" / f"apartments_{date_str}.parquet"
            output_file_path.parent.mkdir(parents=True, exist_ok=True)

            scraper = ImmovlanListingScraper()
            total_scraped = scraper.scrape_listings(urls_txt_file, output_file_path)

            logger.info(
                f"Scraped {total_scraped} apartments, saved to {output_file_path}"
            )
            return str(output_file_path)

        except Exception as e:
            logger.error(f"Apartment scraping failed: {str(e)}")
            raise

    @task
    def scrape_houses() -> str:
        """Scrape house listings and save to parquet."""
        logger = setup_logger(__name__)
        logger.info("Starting house scraping...")

        try:
            urls_txt_file = SITEMAPS_DIR / "houses_links.txt"
            if not urls_txt_file.exists():
                raise FileNotFoundError(f"House links file not found: {urls_txt_file}")

            RAW_DIR.mkdir(parents=True, exist_ok=True)
            date_str = datetime.now(timezone.utc).date()
            output_file_path = RAW_DIR / "houses" / f"houses_{date_str}.parquet"
            output_file_path.parent.mkdir(parents=True, exist_ok=True)

            scraper = ImmovlanListingScraper()
            total_scraped = scraper.scrape_listings(urls_txt_file, output_file_path)

            logger.info(f"Scraped {total_scraped} houses, saved to {output_file_path}")
            return str(output_file_path)

        except Exception as e:
            logger.error(f"House scraping failed: {str(e)}")
            raise

    @task
    def prep_analysis_dataset(apartment_data_path: Any, house_data_path: Any) -> str:
        """Prepare analysis dataset from raw scraped data."""
        logger = setup_logger(__name__)
        logger.info("Preparing analysis dataset...")

        try:
            # Verify input files exist
            if not Path(apartment_data_path).exists():
                raise FileNotFoundError(f"Apartment data not found: {apartment_data_path}")
            if not Path(house_data_path).exists():
                raise FileNotFoundError(f"House data not found: {house_data_path}")

            out_file = prepare_analysis_dataset(Path(apartment_data_path), Path(house_data_path), ANALYSIS_DIR)

            if not out_file.exists():
                raise RuntimeError("Analysis dataset creation failed")

            # Log basic statistics
            df = pd.read_parquet(out_file)
            logger.info(f"Analysis dataset created with {len(df)} records, saved to {out_file}")

            return str(out_file)

        except Exception as e:
            logger.error(f"Analysis data preparation failed: {str(e)}")
            raise

    @task
    def prep_training_dataset(apartment_data_path: Any, house_data_path: Any) -> str:
        """Prepare training and test datasets from raw scraped data."""
        logger = setup_logger(__name__)
        logger.info("Preparing training datasets...")

        try:
            # Verify input files exist
            if not Path(apartment_data_path).exists():
                raise FileNotFoundError(f"Apartment data not found: {apartment_data_path}")
            if not Path(house_data_path).exists():
                raise FileNotFoundError(f"House data not found: {house_data_path}")

            TRAINING_DIR.mkdir(parents=True, exist_ok=True)
            training_dataset_path = prepare_training_dataset(Path(apartment_data_path), Path(house_data_path), TRAINING_DIR)

            if not training_dataset_path.exists():
                raise RuntimeError("Training dataset creation failed")

            # Log basic statistics
            training_df = pd.read_parquet(training_dataset_path)
            logger.info(f"Training dataset: {len(training_df)} records")

            return str(training_dataset_path)

        except Exception as e:
            logger.error(f"Training data preparation failed: {str(e)}")
            raise

    @task
    def train_model(training_dataset_path: Any) -> str:
        """Train regression model using improved training pipeline."""
        logger = setup_logger(__name__)
        logger.info("Starting model training...")

        try:
            if not training_dataset_path or not Path(training_dataset_path).exists():
                raise FileNotFoundError(f"Training data not found: {training_dataset_path}")

            MODELS_DIR.mkdir(parents=True, exist_ok=True)

            # Get model configuration from Airflow Variables (if set)
            try:
                alpha = float(Variable.get("model_alpha", default_var=0.1))
                l1_ratio = float(Variable.get("model_l1_ratio", default_var=0.5))
                test_size = float(Variable.get("model_test_size", default_var=0.2))
            except Exception as e:
                logger.warning(f"Could not load Airflow variables, using defaults: {e}")
                alpha, l1_ratio, test_size = 0.1, 0.5, 0.2

            # Create configuration
            model_config = ModelConfig(
                alpha=alpha, l1_ratio=l1_ratio, test_size=test_size
            )

            trainer = RegressionTrainer(model_config)
            model_path = trainer.train_and_evaluate_model(
                Path(training_dataset_path), MODELS_DIR
            )

            logger.info(f"Model trained successfully, saved to {model_path}")
            return str(model_path)

        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise

    @task
    def cleanup_old_models() -> None:
        """Clean up old model files to save disk space."""
        logger = setup_logger(__name__)
        logger.info("Cleaning up old model files...")

        try:
            # Keep last N models (configurable via Airflow Variable)
            keep_last_n = int(Variable.get("models_keep_last_n", default_var=10))

            model_files = sorted(MODELS_DIR.glob("*.pkl"), reverse=True)

            if len(model_files) > keep_last_n:
                files_to_delete = model_files[keep_last_n:]

                for file_path in files_to_delete:
                    file_path.unlink()
                    logger.info(f"Deleted old model: {file_path}")

                logger.info(f"Cleaned up {len(files_to_delete)} old model files")
            else:
                logger.info(
                    f"No cleanup needed. Found {len(model_files)} models, keeping {keep_last_n}"
                )

        except Exception as e:
            logger.error(f"Model cleanup failed: {str(e)}")
            # Don't fail the entire pipeline for cleanup issues
            pass

    # Define task instances
    t_scrape_sitemaps = scrape_sitemaps()
    t_scrape_apartments = scrape_apartments()
    t_scrape_houses = scrape_houses()
    t_prep_analysis_dataset = prep_analysis_dataset(t_scrape_apartments, t_scrape_houses)
    t_prep_training_dataset = prep_training_dataset(t_scrape_apartments, t_scrape_houses)
    t_train_model = train_model(t_prep_training_dataset)
    t_cleanup = cleanup_old_models()

    # Define task dependencies
    # Both scrapers run in parallel
    t_scrape_sitemaps >> [t_scrape_apartments, t_scrape_houses] >> t_prep_analysis_dataset
    t_scrape_sitemaps >> [t_scrape_apartments, t_scrape_houses] >> t_prep_training_dataset

    # ML pipeline: training -> evaluation -> validation -> cleanup
    t_prep_training_dataset >> t_train_model >> t_cleanup