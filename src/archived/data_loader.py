import boto3
from botocore.config import Config

"""
1. connection 정보 수정
2. bucket, objects 정보 수정
- bucket_name: 조회하는 query에 해당하는 법인의 결과를 저장할 버킷명
- filepath: 쿼리결과 저장할 파일명
- sql
3. dynamo.query() api test
4. s3브라우저나 s3 api로 결과 확인
"""
import time
from io import StringIO
from urllib.parse import urlparse

import pandas as pd

# region
oc_region_name = "ap-northeast-2"
gm1_region_name = "us-east-2"
mi_region_name = "us-east-1"
wa_region_name = "eu-central-1"

# connection 정보(Z:\.aws에서 확인가능)
user_id = "choj289"
aws_access_key_id = "y1KqnptAoG9KZsBN"
aws_secret_access_key = "pNRc2XkmcKF72AFgDX2KF30PoARavzN9Qa3uYU6E"

# connection-S3
oc_s3_endpoint = "http://s3hq.dvc.lgensol.com:9000"
gm1_s3_endpoint = "http://s3.gm1.dvc.lgensol.com:9000"
mi_s3_endpoint = "http://s3.mi.dvc.lgensol.com:9000"
wa_s3_endpoint = "http://s3.wa.dvc.lgensol.com:9000"

# connection-dynamo
oc_dynamo_endpoint = "http://dynamohq.dvc.lgensol.com:9000"
gm1_dynamo_endpoint = "http://dynamo.gm1.dvc.lgensol.com:9000"
mi_dynamo_endpoint = "http://dynamo.mi.dvc.lgensol.com:9000"
wa_dynamo_endpoint = "http://dynamo.wa.dvc.lgensol.com:9000"

retry_config = Config(
    retries={"max_attempts": 1, "mode": "standard"},
    connect_timeout=60 * 60,
    read_timeout=60 * 60,
)


# region_name은 조회하려고 하는 데이터가 존재하는 법인 / endpoint_url은 현재 사용하고 있는 환경의 법인
def S3Client():
    retry_config = Config(
        retries={"max_attempts": 1, "mode": "standard"}, connect_timeout=60 * 600
    )
    s3 = boto3.resource(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=oc_region_name,
        endpoint_url=oc_s3_endpoint,
        config=retry_config,
    )
    return s3


# region_name은 조회하려고 하는 데이터가 존재하는 법인 / endpoint_url은 현재 사용하고 있는 환경의 법인
def dynamoClient():
    client = boto3.client(
        "dynamodb",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=oc_region_name,
        endpoint_url=oc_dynamo_endpoint,
        config=retry_config,
    )

    return client


def escape_partiql_string(value: str) -> str:
    if value is None:
        return "NULL"
    # value = value.replace("'", "''")

    value = value.replace('"', '\\\\"""')
    value = value.replace("`", "\\\\`")
    value = value.replace("'", "''''")

    # 테스트
    value = value.replace("(", "").replace(")", "")

    return value


def load_data(sql: str):
    sql = escape_partiql_string(value=sql)

    s3 = S3Client()

    client = dynamoClient()
    user_id = "choj289"
    bucket_name = "esoc-choj289"  # WA 환경에서 OC 데이터 조회 시 OC 버킷으로 지정 필요
    prefix = "test"  # 버킷하위 경로명
    s3_dir = f"s3://{bucket_name}/{prefix}/test_sql_v4.csv"  # test에 저장할 파일명 변경

    # 쿼리에 해당하는 DB명으로 변경 필요(OC sbpp의 경우 sbpp / WA dvcp의 경우 dvcwap)
    response = client.query(
        TableName=f"""dvcp.glbfnc.dy_sql_startqueryexecution('{user_id}','{s3_dir}','{sql}')"""
    )

    print(response)

    uuid = response["Items"][0]["data"]["S"]
    print("uuid:", uuid)

    # 뭔가 테스트 하는 것 같음
    parsed = urlparse(s3_dir)
    bucket = parsed.netloc
    expected_key = parsed.path.lstrip("/")

    prefix_to_check = "/".join(expected_key.split("/")[:-1])
    if prefix_to_check:
        prefix_to_check += "/"

    print("bucket:", bucket)
    print("expected_key:", expected_key)
    print("prefix_to_check:", prefix_to_check)

    for i in range(60):  # 10초 x 60 = 최대 10분 확인
        print(f"\nCheck {i+1}/60")

        objs = list(s3.Bucket(bucket).objects.filter(Prefix=prefix_to_check))

        if objs:
            print("Objects found:")
            for obj in objs:
                print("-", obj.key, obj.size, obj.last_modified)

            keys = [obj.key for obj in objs]

            if expected_key in keys:
                target_obj = [
                    obj for obj in objs if obj.key == expected_key][0]
                if target_obj.size > 0:
                    print("\n✅ Expected result file exists and size > 0")
                    break
                else:
                    print("\nExpected file exists but size is 0. May still be writing.")

            csv_keys = [
                obj.key for obj in objs if obj.key.lower().endswith(".csv")]
            if csv_keys:
                print("\n✅ CSV file found:", csv_keys[0])
                break

        else:
            print("No result file yet.")

        time.sleep(10)
    else:
        print("\n⚠️ No result file found within timeout.")

    # 로드
    key = "test/test_sql_v4.csv"

    obj = s3.Object(bucket, key).get()
    body = obj["Body"].read().decode("utf-8")

    df = pd.read_csv(StringIO(body))
    return df
