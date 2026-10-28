"""Download pinned model revisions and framework datasets directly to this workspace."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

from huggingface_hub import snapshot_download

COMMIT = '65819e14a49c940fa7b2977f07872cabd6229129'
DATA = {
    'train/deepscaler_conversation.jsonl': '01fea8ef9b43a8edf0f7fe265dde02b01697cc453c220d33e394398f3c02ffc4',
    'eval/math_test.jsonl': '36d09ef3bcb0018398cd44f6206b578e0939434ff974be321fbbfcc4bc17b2de',
}
MODELS = {
    'Qwen/Qwen3-0.6B-Base': 'da87bfb608c14b7cf20ba1ce41287e8de496c0cd',
    'Qwen/Qwen3-4B': '1cfa9a7208912126459214e8b04321603b3df60c',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models-dir', type=Path, default=Path('models'))
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--skip-models', action='store_true')
    args = parser.parse_args()
    for relative, expected in DATA.items():
        output = args.data_dir/relative
        if output.exists() and hashlib.sha256(output.read_bytes()).hexdigest() == expected:
            print(f'Already verified: {output}')
            continue
        if output.exists():
            raise ValueError(f'Refusing to overwrite an unexpected existing file: {output}')
        url = f'https://raw.githubusercontent.com/EIT-NLP/PowerOPD/{COMMIT}/data/{relative}'
        with urllib.request.urlopen(url, timeout=120) as source:
            content = source.read()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f'Dataset hash mismatch: {url}')
        output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation avoids silently overwriting a concurrent edit.
        with output.open('xb') as target:
            target.write(content)
        print(f'Downloaded and verified: {output}')
    if not args.skip_models:
        for model_id, revision in MODELS.items():
            snapshot_download(
                repo_id=model_id, revision=revision,
                local_dir=args.models_dir/model_id.split('/')[-1],
                cache_dir=args.models_dir/'.hub-cache',
                allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', '*.model'],
            )
    print(json.dumps({'data_source_commit': COMMIT, 'model_revisions': MODELS}, indent=2))


if __name__ == '__main__':
    main()
