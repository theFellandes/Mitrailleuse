from mitrailleuse.infrastructure.utils.similarity_checker import SimilarityChecker

def test_similarity_checker(tmp_path):
    config = {
        "general": {
            "similarity_check": {
                "enabled": True,
                "settings": {
                    "similarity_threshold": 0.8,
                    "cooldown_period": 1,
                    "max_recent_responses": 2,
                    "close_after_use": True
                }
            }
        }
    }
    checker = SimilarityChecker(tmp_path, config)
    is_similar, sim = checker.check_similarity({"choices": [{"message": {"content": "test"}}]})
    assert not is_similar
    checker.close()
