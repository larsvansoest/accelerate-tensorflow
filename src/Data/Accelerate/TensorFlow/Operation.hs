{-# LANGUAGE BangPatterns      #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs             #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies      #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Use record patterns" #-}

module Data.Accelerate.TensorFlow.Operation where

import Data.Array.Accelerate.AST.Operation
import Data.Array.Accelerate.AST.LeftHandSide
import Data.Array.Accelerate.Backend
import Data.Array.Accelerate.Type
import Data.Array.Accelerate.Representation.Array
import Data.Array.Accelerate.Representation.Type
import Data.Array.Accelerate.Trafo.Var
import Data.Array.Accelerate.AST.Idx
import Data.Array.Accelerate.Trafo.Operation.Substitution
import Data.Array.Accelerate.AST.Environment
import Data.Array.Accelerate.Representation.Ground
import Data.Array.Accelerate.Trafo.Desugar
import Data.Array.Accelerate.Trafo.Exp.Substitution
import Data.Type.Equality
import Data.Array.Accelerate.Lifetime
import Foreign
import Data.Array.Accelerate.AST.Kernel (NoKernelMetadata)
import Data.Text.Prettyprint.Doc
import Data.Array.Accelerate.Pretty.Exp
    ( prettyConst, primOperator )
import Data.Array.Accelerate.Pretty.Print (Operator(..))
import Data.Array.Accelerate.Pretty.Type (prettyScalarType)
import Data.Array.Accelerate.Representation.Shape

newtype BufferIdx benv a = BIdx (Idx benv (Buffer a)) -- a is enkele binding uit env
-- | Environment with indexes pointing to buffers

-- benv is buffer env
-- env is scalar env
-- in env staat een int, en en benv kan ik op zoek naar een buffer van ints
type BufferEnv benv env = Env (BufferIdx benv) env

data TensorOp op where
  TNil :: TensorOp ()
  TConst :: ScalarType s -> s -> TensorOp (Out sh s -> ())
  TPrimFun :: PrimFun (a -> b) -> TensorOp (In sh a -> Out sh b -> ())
  TVar :: ScalarType s -> Idx env (Buffer s) -> TensorOp (Out sh s -> ())

instance PrettyOp TensorOp where
  prettyOp TNil         = "TNil"
  prettyOp (TConst s e) = vsep ["TConst", prettyConst (TupRsingle s) e]
  prettyOp (TPrimFun f) = vsep ["TBinOp", opName (primOperator f) ]
  prettyOp (TVar _ _)   = "TVar"

instance NFData' TensorOp where
  rnf' !_ = ()

-- forall is alleen nodig als je @s wilt gebruiken in de method definition
f :: forall env s. TypeR s -> TupR (Var GroundR env) (Distribute Buffer s) -> Distribute (BufferIdx env) s
f TupRunit _ = ()
f (TupRsingle s) (TupRsingle (Var _ idx)) 
  | Refl <- reprIsSingle @ScalarType @s @(BufferIdx env) s
  , Refl <- reprIsSingle @ScalarType @s @Buffer s
  = BIdx idx
f (TupRpair l1 l2) (TupRpair r1 r2) = (f l1 r1, f l2 r2)
f _ _ = error "impossible"

instance DesugarAcc TensorOp where
  mkMap (ArgFun (Lam lhs (Body body))) (ArgArray _ (ArrayR _ t) _ gvb) aOut =
    mkMapF (push' Empty (lhs, f t gvb)) body aOut
  mkMap _ _ _ = error "impossible"

  mkGenerate f aOut@(ArgArray _ (ArrayR sh _) gv _)
    | sh' <- shapeType sh
    , DeclareVars lhs w k <- declareVars $ buffersR sh'
    = aletUnique lhs (desugarAlloc (ArrayR sh sh') (fromGrounds gv)) $
      mkMap (weaken w f) (ArgArray In (ArrayR sh sh') (weakenVars w gv) (k weakenId)) (weaken w aOut)

  -- The result array is initialised with the given defaults and 
  -- any further values that are permuted into the result array 
  -- are added to the current value using the given combination function.

  -- The combination function is given the new value being permuted as its first argument, 
  -- and the current value of the array as its second.
  mkPermute comb defaults perm source
    = undefined






-- lookupBIEnv :: Idx env' t -> BIEnv env env' -> Idx env (Buffer t)
-- lookupBIEnv ZeroIdx (Push _ (BIdx bidx)) = bidx
-- lookupBIEnv (SuccIdx idx) (Push bidxs _) = lookupBIEnv idx bidxs

mkMapF :: forall env env' sh t. BufferEnv env env' -> PreOpenExp (ArrayInstr env) env' t
  -> Arg env (Out sh t) -> OperationAcc TensorOp env ()
mkMapF _ (Const s e) aOut = Exec (TConst s e) $ aOut :>: ArgsNil

mkMapF env (PrimApp f exp) aOut@(ArgArray _ (ArrayR sh _) gv _)
 | a <- expType exp
 , DeclareVars lhs w k <- declareVars $ buffersR a
 = aletUnique lhs (desugarAlloc (ArrayR sh a) (fromGrounds gv)) $
   Alet (LeftHandSideWildcard TupRunit) TupRunit
   (mkMapF
     (weakenBufferEnv w env)
     (weakenArrayInstr w exp)
     (ArgArray Out (ArrayR sh a) (weakenVars w gv) (k weakenId))
   ) $
   Exec
    (TPrimFun f)
    (ArgArray In
      (ArrayR sh a)
      (weakenVars w gv)
      (k weakenId) :>: weaken w aOut :>: ArgsNil
    )

mkMapF env (Let elhs exp1 exp2) aOut@(ArgArray _ (ArrayR sh _) gv _)
 | a <- expType exp1
 , DeclareVars lhs w k <- declareVars $ buffersR a
 = aletUnique lhs (desugarAlloc (ArrayR sh a) (fromGrounds gv)) $
   Alet (LeftHandSideWildcard TupRunit) TupRunit
   (mkMapF
     (weakenBufferEnv w env)
     (weakenArrayInstr w exp1)
     (ArgArray Out (ArrayR sh a) (weakenVars w gv) (k weakenId))
   ) $
   mkMapF
     (pushBufferEnv elhs (k weakenId) (weakenBufferEnv w env))
     (weakenArrayInstr w exp2)
     (weaken w aOut)

mkMapF env (Evar (Var s idx)) aOut = let
  x = prj idx env in
  Exec (TVar s) $ _

mkMapF env (Pair exp1 exp2)
  (ArgArray _ (ArrayR sh (TupRpair t1 t2)) gv (TupRpair gvb1 gvb2))
 = Alet (LeftHandSideWildcard TupRunit) TupRunit (
    mkMapF env exp1 (ArgArray Out (ArrayR sh t1) gv gvb1)) $
   mkMapF env exp2 (ArgArray Out (ArrayR sh t2) gv gvb2)

-- TODO

mkMapF _ (Foreign _ _ _ _) _ = undefined
mkMapF _ Nil _ = undefined
mkMapF _ (VecPack _ _) _ = undefined
mkMapF _ (VecUnpack _ _) _ = undefined
mkMapF _ (IndexSlice _ _ _) _ = undefined
mkMapF _ (IndexFull _ _ _) _ = undefined
mkMapF _ (ToIndex _ _ _) _ = undefined
mkMapF _ (FromIndex _ _ _) _ = undefined
mkMapF _ (Case _ _ _) _ = undefined
mkMapF _ (Cond _ _ _) _ = undefined
mkMapF _ (While _ _ _) _ = undefined
mkMapF _ (PrimConst _) _ = undefined
mkMapF _ (ArrayInstr _ _) _ = undefined
mkMapF _ (ShapeSize _ _) _ = undefined
mkMapF _ (Undef _) _ = undefined
mkMapF _ (Coerce _ _ _) _ = undefined
mkMapF _ _ _ = error "impossible"

-- temp kernel for testing purposes

data TensorFlowKernel env where
  TensorFlowKernel
    :: { kernelId       :: Int
       , kernelFunction :: !(Lifetime (FunPtr  env))
       }
    -> TensorFlowKernel env

instance NFData' TensorFlowKernel where
  rnf' (TensorFlowKernel !_ fn) = unsafeGetValue fn `seq` ()

newtype TensorFlowKernelMetadata f =
  TensorFlowKernelMetadata { kernelArgsSize :: Int }

instance IsKernel TensorFlowKernel where
  type KernelOperation TensorFlowKernel = TensorOp
  type KernelMetadata  TensorFlowKernel = NoKernelMetadata
  compileKernel = undefined

instance PrettyKernel TensorFlowKernelMetadata where
  prettyKernel = PrettyKernelBody True $ \_ kernel -> ""
