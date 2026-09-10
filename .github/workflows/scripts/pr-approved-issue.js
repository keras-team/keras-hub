/** Enforce that external PRs link an approved issue assigned to the author.
 *
 *  - On `opened`, the PR is converted to a draft and an "Approved issue link"
 *    section is added to the description if it is missing.
 *  - On every run, the description is scanned for issue references
 *    (`#123`, `owner/repo#123` or a full issue URL). The PR passes when at
 *    least one referenced issue exists in this repo and has the PR author as
 *    an assignee.
 *  - When the check passes, a draft PR is marked "Ready for review". When it
 *    fails and the PR is not a draft, it is converted to a draft. A single
 *    sticky comment is kept up to date with the result.
 *
 *  Maintainers, collaborators and bots are skipped by the workflow `if:`.
 */

const SECTION_HEADING = "## Approved issue link";
const SECTION_TEMPLATE = `${SECTION_HEADING}
<!--- Link the approved issue that is assigned to you, e.g. "Fixes #123".
      An issue must be assigned to you and linked here before this PR can be
      marked "Ready for review". -->

`;
const COMMENT_MARKER = "<!-- pr-approved-issue-check -->";
const BYPASS_ASSOCIATIONS = ["OWNER", "MEMBER", "COLLABORATOR"];

module.exports = async ({ github, context, core }) => {
  const pr = context.payload.pull_request;
  if (!pr) {
    console.log("Not a pull request payload.");
    return;
  }

  const { owner, repo } = context.repo;
  const action = context.payload.action;
  const author = pr.user.login;

  // Defence in depth: the workflow `if:` already filters these out.
  if (pr.user.type === "Bot" || BYPASS_ASSOCIATIONS.includes(pr.author_association)) {
    console.log(`Skipping #${pr.number}: ${author} is ${pr.author_association}.`);
    return;
  }

  let body = pr.body || "";
  let isDraft = pr.draft;

  // 1. On open: make sure the description has the section and start as draft.
  if (action === "opened") {
    if (!body.includes(SECTION_HEADING)) {
      body = SECTION_TEMPLATE + body;
      await github.rest.pulls.update({ owner, repo, pull_number: pr.number, body });
      console.log(`Added "${SECTION_HEADING}" section to #${pr.number}.`);
    }
    if (!isDraft) {
      isDraft = await convertToDraft(github, core, pr);
    }
  }

  // 2. Find issues referenced in the description that are assigned to the author.
  const referenced = findIssueNumbers(body, owner, repo);
  const assigned = [];
  const notAssigned = [];
  for (const number of referenced) {
    try {
      const { data: issue } = await github.rest.issues.get({ owner, repo, issue_number: number });
      if (issue.pull_request) continue; // A PR reference, not an issue.
      const assignees = (issue.assignees || []).map((a) => a.login.toLowerCase());
      (assignees.includes(author.toLowerCase()) ? assigned : notAssigned).push(number);
    } catch (err) {
      console.log(`Could not fetch issue #${number}: ${err.message}`);
    }
  }
  const passed = assigned.length > 0;

  // 3. Flip the draft state to match the result.
  if (passed && isDraft) {
    isDraft = await markReadyForReview(github, core, pr);
  } else if (!passed && !isDraft) {
    isDraft = await convertToDraft(github, core, pr);
  }

  // 4. Report via a sticky comment and the job status.
  const message = passed
    ? [
        `✅ Approved issue check passed: ${assigned.map((n) => `#${n}`).join(", ")} ` +
          `is assigned to @${author}.`,
        isDraft
          ? "Could not mark this PR as ready automatically; please mark it **Ready for review**."
          : "This PR is **Ready for review**.",
      ]
    : [
        `❌ Approved issue check failed. This PR ${isDraft ? "stays" : "must stay"} in **draft** until ` +
          "it links an approved issue that is assigned to you.",
        "",
        "To fix this:",
        "1. Find or open an issue for this change and ask a maintainer to approve it and assign it to you.",
        `2. Link it under the "${SECTION_HEADING.replace("## ", "")}" section of this PR's description, e.g. \`Fixes #123\`.`,
        "",
        "The PR will be marked **Ready for review** automatically once the check passes.",
        "",
        notAssigned.length
          ? `Referenced issue(s) not assigned to @${author}: ${notAssigned.map((n) => `#${n}`).join(", ")}.`
          : "No issue reference was found in the description.",
      ];
  await upsertComment(github, owner, repo, pr.number, message.join("\n"));

  if (!passed) {
    core.setFailed(`No approved issue assigned to ${author} is linked in #${pr.number}.`);
  }
};

/** Collect issue numbers referenced in `body` that belong to this repo. */
function findIssueNumbers(body, owner, repo) {
  const escaped = `${owner}/${repo}`.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const patterns = [
    new RegExp(`https?://github\\.com/${escaped}/issues/(\\d+)`, "gi"),
    new RegExp(`(?:^|[^\\w/])${escaped}#(\\d+)`, "gi"),
    // Bare `#123`, but not `owner/repo#123` of some other repo.
    /(?:^|[^\w/])#(\d+)\b/g,
  ];
  const numbers = new Set();
  for (const pattern of patterns) {
    for (const match of body.matchAll(pattern)) {
      numbers.add(Number(match[1]));
    }
  }
  return [...numbers];
}

async function convertToDraft(github, core, pr) {
  try {
    await github.graphql(
      `mutation($id: ID!) {
        convertPullRequestToDraft(input: {pullRequestId: $id}) {
          pullRequest { isDraft }
        }
      }`,
      { id: pr.node_id }
    );
    console.log(`Converted #${pr.number} to draft.`);
    return true;
  } catch (err) {
    warnDraftToggleFailed(core, `convert #${pr.number} to draft`, err);
    return pr.draft;
  }
}

async function markReadyForReview(github, core, pr) {
  try {
    await github.graphql(
      `mutation($id: ID!) {
        markPullRequestReadyForReview(input: {pullRequestId: $id}) {
          pullRequest { isDraft }
        }
      }`,
      { id: pr.node_id }
    );
    console.log(`Marked #${pr.number} as ready for review.`);
    return false;
  } catch (err) {
    warnDraftToggleFailed(core, `mark #${pr.number} as ready for review`, err);
    return true;
  }
}

function warnDraftToggleFailed(core, what, err) {
  const lines = [`Could not ${what}: ${err.message}`];
  if (/not accessible by integration/i.test(err.message)) {
    lines.push("Toggling draft state needs `contents: write` and `pull-requests: write`.");
  }
  core.warning(lines.join("\n"));
}

async function upsertComment(github, owner, repo, issue_number, text) {
  const body = `${COMMENT_MARKER}\n${text}`;
  const comments = await github.paginate(github.rest.issues.listComments, {
    owner,
    repo,
    issue_number,
    per_page: 100,
  });
  const existing = comments.find((c) => c.body && c.body.startsWith(COMMENT_MARKER));
  if (existing) {
    if (existing.body !== body) {
      await github.rest.issues.updateComment({ owner, repo, comment_id: existing.id, body });
    }
  } else {
    await github.rest.issues.createComment({ owner, repo, issue_number, body });
  }
}
