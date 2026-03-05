from structured_knowledge import build_structured_knowledge


def main():
    data = build_structured_knowledge()
    print("[OK] Structured extraction completed")
    print(f"Decisions: {len(data.get('decisions', []))}")
    print(f"Rules: {len(data.get('rules', []))}")
    print(f"Warnings: {len(data.get('warnings', []))}")


if __name__ == "__main__":
    main()
