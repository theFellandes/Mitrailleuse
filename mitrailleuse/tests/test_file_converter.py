from mitrailleuse.infrastructure.utils.file_converter import FileConverter
import json

def test_backup_and_convert(tmp_path):
    # Create a JSON file
    data = [{"a": 1}, {"a": 2}]
    json_file = tmp_path / "test.json"
    json_file.write_text(json.dumps(data))
    # Convert to JSONL
    jsonl_file = FileConverter.convert_to_jsonl(json_file, tmp_path)
    assert jsonl_file.exists()
    # Convert back to JSON
    json_file2 = FileConverter.convert_to_json(jsonl_file, tmp_path)
    assert json_file2.exists()
    # Backup should exist
    backup_dir = tmp_path / "inputs" / "backup"
    assert any(backup_dir.iterdir())
