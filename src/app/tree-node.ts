export class TreeNode {
  val: number
  left: TreeNode | null
  right: TreeNode | null

  constructor(val?: number, left?: TreeNode | null, right?: TreeNode | null) {
    this.val = (val === undefined ? 0 : val)
    this.left = (left === undefined ? null : left)
    this.right = (right === undefined ? null : right)
  }
}

//[5,4,8,11,null,13,4,7,2,null,null,null,1]
export const convertTree = (items: any[]): TreeNode | null => {
  if(items.length == 0) {
    return null;
  }
  let root = new TreeNode(items.shift());
  let stack: any[] = [root];
  while(items.length > 0) {
    let nextItems = [];
    while(stack.length > 0){
      let node = stack.shift();
      let left = items.shift();
      node.left = left != null ? new TreeNode(left) : null;
      if(node.left != null) {
        nextItems.push(node.left);
      }
      let right = items.shift();
      node.right = right != null ? new TreeNode(right) : null;
      if(node.right) {
        nextItems.push(node.right);
      }
    }
    stack = nextItems;

  }

  return root;
}
