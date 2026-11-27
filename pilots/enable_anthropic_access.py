"""Turn on Anthropic models (our judge) for one AWS account. Run once, by the account owner.

Claude Haiku is the judge in grader/judge.py. Before any Anthropic model answers, AWS needs two
things from the account, and both are account-wide rather than per user:

  1. a use case form, whose contents are passed on to Anthropic
  2. acceptance of the model agreement for the model we use

The old "Model access" page in the Bedrock console has been retired, so this is now done through the
API. Nothing here needs console access: the keys you already use for Bedrock are enough.

**Who should run this.** The person who owns the account. It states a company, an industry and an
intended use on the account's behalf and accepts terms on it. If you are using someone else's keys,
send them this file rather than running it yourself.

    python3 pilots/enable_anthropic_access.py          # shows what it would do, changes nothing
    python3 pilots/enable_anthropic_access.py --apply  # asks you to confirm, then does it

Access usually works within about 15 minutes. Check with:

    python3 pilots/check_bedrock.py
"""
import argparse
import json
import sys

sys.path.insert(0, __file__.rsplit("/pilots/", 1)[0])

from models.settings import JUDGE_MODEL_ID  # noqa: E402

# ─── Fill these in with the ACCOUNT OWNER'S real details. They are sent to Anthropic. ───
FORM = {
    "companyName": "",            # required, up to 128 characters
    "companyWebsite": "",         # required, up to 128 characters
    "intendedUsers": "0",         # "0" internal, "1" external, "2" both — a number, not words
    "industryOption": "Education",
    "otherIndustryOption": "",    # only when industryOption is "Other"
    "useCases": "Academic research evaluating how language models follow instructions "
                "across a mid-conversation language switch.",
}


def main():
    parser = argparse.ArgumentParser(prog="enable_anthropic_access")
    parser.add_argument("--apply", action="store_true", help="actually submit, after confirming")
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--model", default=JUDGE_MODEL_ID)
    args = parser.parse_args()

    import boto3
    bedrock = boto3.client("bedrock", region_name=args.region)

    missing = [k for k in ("companyName", "companyWebsite") if not FORM[k].strip()]
    if missing:
        sys.exit(f"Fill in {', '.join(missing)} at the top of this file first. "
                 f"They go to Anthropic, so they should be the account's real details.")

    try:
        bedrock.get_use_case_for_model_access()
        print("The use case form is already on this account.")
        form_needed = False
    except bedrock.exceptions.ResourceNotFoundException:
        print("The use case form has not been submitted for this account.")
        form_needed = True

    offers = bedrock.list_foundation_model_agreement_offers(modelId=args.model).get("offers", [])
    if not offers:
        sys.exit(f"No agreement offer found for {args.model} in {args.region}.")
    offer_token = offers[0]["offerToken"]

    print()
    print("About to do the following for this AWS account:")
    if form_needed:
        print("  1. submit this use case form, which AWS passes on to Anthropic:")
        for key, value in FORM.items():
            print(f"       {key}: {value}")
    print(f"  {'2' if form_needed else '1'}. accept the model agreement for {args.model}")
    print()

    if not args.apply:
        print("Nothing was changed. Re-run with --apply to do it.")
        return

    print("This accepts terms on behalf of the account owner.")
    if input("Type the word yes to go ahead: ").strip().lower() != "yes":
        sys.exit("Stopped. Nothing was changed.")

    if form_needed:
        bedrock.put_use_case_for_model_access(formData=json.dumps(FORM).encode("utf-8"))
        print("  use case form submitted")
    bedrock.create_foundation_model_agreement(modelId=args.model, offerToken=offer_token)
    print("  model agreement accepted")
    print()
    print("Access can take about 15 minutes. Then check with:")
    print("    python3 pilots/check_bedrock.py")


if __name__ == "__main__":
    main()
