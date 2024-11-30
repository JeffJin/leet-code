export class TreeNode {
  val: number
  left: TreeNode | null;
  right: TreeNode | null;
  parent: TreeNode | null;

  constructor(val?: number, left?: TreeNode | null, right?: TreeNode | null, parent?: TreeNode | null) {
    this.val = (val === undefined ? 0 : val)
    this.left = (left === undefined ? null : left)
    this.right = (right === undefined ? null : right)
    this.parent = (parent === undefined ? null : parent)
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
      let leftVal = items.shift();
      node.left = leftVal != null ? new TreeNode(leftVal) : null;
      if(node.left != null) {
        nextItems.push(node.left);
      }
      let rightVal = items.shift();
      node.right = rightVal != null ? new TreeNode(rightVal) : null;
      if(node.right != null) {
        nextItems.push(node.right);
      }
    }
    stack = nextItems;

  }

  return root;
}
