from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime, timedelta
from docker.types import Mount
import os
import json

HOST_PROJECT_PATH = os.getenv('HOST_PROJECT_PATH', 'D:/VSCodeProjects/mlops_lab_1')
R2_THRESHOLD = float(os.getenv('R2_THRESHOLD', '0.70'))

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2026, 3, 24),
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

def check_model_quality():
    """Перевіряє metrics.json і направляє DAG по відповідній гілці"""
    metrics_path = '/opt/airflow/project/metrics.json'
    
    if not os.path.exists(metrics_path):
        print("Файл metrics.json не знайдено!")
        return 'model_rejected'
    
    with open(metrics_path, 'r', encoding="utf-8") as f:
        metrics = json.load(f)
    
    r2_score = metrics.get('r2', 0)
    print(f"Model R2 Score: {r2_score}")
    
    if r2_score >= R2_THRESHOLD:
        return 'model_accepted'
    else:
        return 'model_rejected'

with DAG(
    'house_rent_mlops_pipeline',
    default_args=default_args,
    description='House Rent Pipeline: Download, Prepare, Train, Test',
    schedule_interval=timedelta(days=1),
    catchup=False,
) as dag:

    venv_mount = Mount(source="", target='/app/.venv', type='volume')
    project_mount = Mount(source=HOST_PROJECT_PATH, target='/app', type='bind')

    docker_kwargs = {
        'image': 'mlops-lab-model:latest',
        'docker_url': 'unix://var/run/docker.sock',
        'network_mode': 'mlops-network',
        'auto_remove': 'force',
        'mount_tmp_dir': False,
        'mounts': [project_mount, venv_mount],
        'environment': {
            'MLFLOW_TRACKING_URI': 'http://mlflow-server:5000',
            'UV_LINK_MODE': 'copy',
            'R2_THRESHOLD': str(R2_THRESHOLD)
        }
    }

    check_status = DockerOperator(
        task_id='check_dvc_status',
        command='rm -f .dvc/tmp/rwlock && uv run dvc status',
        **docker_kwargs
    )

    download_data = DockerOperator(
        task_id='download_data',
        command='uv run dvc repro download',
        **docker_kwargs
    )

    prepare_data = DockerOperator(
        task_id='prepare_data',
        command='uv run dvc repro prepare',
        **docker_kwargs
    )

    test_data = DockerOperator(
        task_id='test_data',
        command='uv run dvc repro test_data',
        **docker_kwargs
    )

    train_model = DockerOperator(
        task_id='train_model',
        command='uv run dvc repro train',
        **docker_kwargs
    )

    test_model = DockerOperator(
        task_id='test_model',
        command='uv run dvc repro test_model',
        **docker_kwargs
    )

    branching = BranchPythonOperator(
        task_id='evaluate_quality_gate',
        python_callable=check_model_quality,
        trigger_rule=TriggerRule.ALL_DONE
    )

    model_accepted = EmptyOperator(task_id='model_accepted')
    model_rejected = EmptyOperator(task_id='model_rejected')

    (
        check_status 
        >> download_data 
        >> prepare_data 
        >> test_data 
        >> train_model 
        >> test_model 
        >> branching
    )
    branching >> [model_accepted, model_rejected]