from src.utils.google_auth import get_credentials, ALL_SCOPES, TOKEN_FILE

def main():
    print("Bootstrapping Gmail + Calendar scopes...")
    creds = get_credentials(ALL_SCOPES)
    print("Saved:", TOKEN_FILE)
    print("Granted scopes:", creds.scopes)
if __name__ == "__main__":
    main()
