"""New public interface: read-only Replay and explicitly separate trusted setup."""

import argparse
import json
from pathlib import Path

from public_replay.bundle import Bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        help="Reviewed public data directory; default: installed package resources",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("replay")
    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("--release-root", type=Path)
    demo = commands.add_parser("demo")
    demo.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("check")
    check.add_argument("--notebook", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    check.add_argument("--forbid-root", action="append", default=[])
    doctor_parser = commands.add_parser("doctor")
    doctor_parser.add_argument("--resource-root", type=Path)
    doctor_parser.add_argument("--check-docker", action="store_true")
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--project-root", type=Path, required=True)
    prepare_parser.add_argument("--resource-root", type=Path, required=True)
    prepare_parser.add_argument(
        "--upstream-url", default="https://github.com/langchain-ai/langchain.git"
    )
    prepare_parser.add_argument("--task", default="LC-INT-TEMPERATURE-001")
    prepare_parser.add_argument("--controls", action="store_true")
    args = parser.parse_args()
    bundle = Bundle(args.data)
    if args.command == "validate":
        from public_replay.validate import validate_release

        if args.release_root:
            validate_release(args.release_root)
        result = {"status": "validated", "content_id": bundle.manifest["content_id"]}
    elif args.command == "replay":
        from public_replay.analysis import comparison, operational

        result = {
            "six_task": json.loads(comparison(bundle).to_json(orient="records")),
            "earlier_two_task": json.loads(
                comparison(bundle, "held_out_two").to_json(orient="records")
            ),
            "operational": json.loads(operational(bundle).to_json(orient="records")),
            "optional_audit": bundle.optional_audit(),
        }
    elif args.command == "demo":
        from public_replay.demo import build_demo

        if args.output.exists():
            raise FileExistsError("Demo output must be a new directory")
        build_demo(bundle.root, args.output)
        result = {"status": "saved", "mode": "replay", "output": str(args.output)}
    elif args.command == "check":
        from public_replay.validate import validate_demo

        result = validate_demo(
            bundle.root, args.notebook, args.output, args.forbid_root
        )
    elif args.command == "doctor":
        from experiments.portable_prepare import doctor

        result = doctor(Path(__file__).resolve().parents[1])
    else:
        from public_preparation.runtime import prepare, validate

        project = Path(__file__).resolve().parents[1]
        expected = project / ".local/public-v1" / args.task
        if (
            args.project_root.resolve() != project
            or args.resource_root.resolve() != expected
        ):
            raise ValueError(
                "Use this checkout and its ignored .local/public-v1/TASK directory for public preparation"
            )
        if args.upstream_url != "https://github.com/langchain-ai/langchain.git":
            raise ValueError(
                "The public protocol requires its pinned upstream repository"
            )
        result = prepare(args.task, network=True)
        if args.controls:
            result = validate(args.task)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
