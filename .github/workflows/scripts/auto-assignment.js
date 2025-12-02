/** Automatically assign issues and PRs to users in the `assigneesList` 
 *  on a rotating basis.
*/

module.exports = async ({ github, context }) => {
  const isIssue = !!context.payload.issue;
  const isPr = !!context.payload.pull_request;
  
  let candidates = [];
  let itemNumber;
  let author;

  // 1. Determine if this is an Issue or PR and select the list
  if (isIssue) {
    candidates = ["dhantule", "sachinprasadhs", "LakshmiKalaKadali", "maitry63"];
    itemNumber = context.payload.issue.number;
    author = context.payload.issue.user.login;
  } else if (isPr) {
    // Reviewer list for PRs
    candidates = ["sachinprasadhs", "divyashreepathihalli"];
    itemNumber = context.payload.pull_request.number;
    author = context.payload.pull_request.user.login;
  } else {
    console.log("Not an issue or PR payload.");
    return;
  }

  if (!candidates.length) {
    console.log("No candidates found.");
    return;
  }

  // 2. Select a candidate based on the Issue/PR number (Rotation)
  let selectionIndex = itemNumber % candidates.length;
  let selectedUser = candidates[selectionIndex];

  // 3. No author restriction: allow assigning any candidate (including the PR author)
  if (isPr) {
    console.log(`Processing #${itemNumber}. Reviewers: ${candidates.join(', ')}`);
  } else {
    console.log(`Processing #${itemNumber}. Selected: ${selectedUser}`);
  }

  // 4. Perform the action
  try {
    if (isPr) {
      // For PRs: Request Reviews from all reviewers in the PR list
      await github.rest.pulls.requestReviewers({
        owner: context.repo.owner,
        repo: context.repo.repo,
        pull_number: itemNumber,
        reviewers: candidates,
      });
    } else {
      // For Issues: Add an Assignee
      await github.rest.issues.addAssignees({
        issue_number: itemNumber,
        owner: context.repo.owner,
        repo: context.repo.repo,
        assignees: [selectedUser],
      });
    }
  } catch (err) {
    console.log("Error adding assignee/reviewer:", err.message);
  }
};