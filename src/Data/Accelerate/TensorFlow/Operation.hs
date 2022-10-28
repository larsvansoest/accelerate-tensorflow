{-# LANGUAGE BangPatterns      #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs             #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies      #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}

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

data TensorOp op where
  TConst :: ScalarType s -> s -> TensorOp (Out sh s -> ())
  TPrimFun :: PrimFun (a -> b) -> TensorOp (In sh a -> Out sh b -> ())
  TVar :: ScalarType s -> Idx env (Buffer s) -> TensorOp (Out sh s -> ())

instance PrettyOp TensorOp where
  prettyOp (TConst s _) = "TTensor"
  prettyOp (TPrimFun _) = "TBinOp"
  prettyOp (TVar _ _)   = "TVar"

instance NFData' TensorOp where
  rnf' !_ = ()

instance DesugarAcc TensorOp where
  mkMap (ArgFun (Lam lhs (Body body))) (ArgArray _ _ _ gvb) aOut = mkMapF (addLHSToBIEnv Empty lhs gvb) body aOut 
  mkMap _ _ _ = error "impossible"

  mkGenerate = undefined
  mkPermute = undefined

data BIdx env a where
  BIdx  :: Idx env (Buffer a) -> BIdx env a

instance Sink BIdx where
  weaken w (BIdx idx) = BIdx $ weaken w idx

type BIEnv env = Env (BIdx env)

weakenBIEnv :: forall benv benv' env. benv :> benv' -> BIEnv benv env -> BIEnv benv' env
weakenBIEnv w = mapEnv (weaken w)

lookupBIEnv :: Idx env' t -> BIEnv env env' -> Idx env (Buffer t)
lookupBIEnv ZeroIdx (Push _ (BIdx bidx)) = bidx
lookupBIEnv (SuccIdx idx) (Push bidxs _) = lookupBIEnv idx bidxs

addLHSToBIEnv :: forall a env env' env'' sh.
  BIEnv env env'
  -> ELeftHandSide a env' env''
  -> TupR (Var GroundR env) (Buffers a) 
  -> BIEnv env env''
addLHSToBIEnv env (LeftHandSideSingle s) (TupRsingle (Var _ idx))
  | Refl <- reprIsSingle @ScalarType @a @Buffer s
  = Push env (BIdx idx)
addLHSToBIEnv env (LeftHandSidePair l1 l2) (TupRpair t1 t2) = addLHSToBIEnv (addLHSToBIEnv env l1 t1) l2 t2
addLHSToBIEnv env (LeftHandSideWildcard _) _ = env
addLHSToBIEnv _ _ _ = error "impossible"

mkMapF :: forall env env' sh t. BIEnv env env' -> PreOpenExp (ArrayInstr env) env' t -> Arg env (Out sh t) -> OperationAcc TensorOp env ()
mkMapF _ (Const s e) aOut = Exec (TConst s e) $ aOut :>: ArgsNil

mkMapF env (PrimApp f exp) aOut@(ArgArray _ (ArrayR sh _) gv _)
 | a <- expType exp
 , DeclareVars lhs w k <- declareVars $ buffersR a
 = aletUnique lhs (desugarAlloc (ArrayR sh a) (fromGrounds gv)) $
   Alet (LeftHandSideWildcard TupRunit) TupRunit (mkMapF (weakenBIEnv w env) (weakenArrayInstr w exp) (ArgArray Out (ArrayR sh a) (weakenVars w gv) (k weakenId))) $
   Exec (TPrimFun f) (ArgArray In (ArrayR sh a) (weakenVars w gv) (k weakenId) :>: weaken w aOut :>: ArgsNil)

mkMapF env (Let elhs exp1 exp2) aOut@(ArgArray _ (ArrayR sh _) gv _)
 | a <- expType exp1
 , DeclareVars lhs w k <- declareVars $ buffersR a
 = aletUnique lhs (desugarAlloc (ArrayR sh a) (fromGrounds gv)) $
   Alet (LeftHandSideWildcard TupRunit) TupRunit (mkMapF (weakenBIEnv w env) (weakenArrayInstr w exp1) (ArgArray Out (ArrayR sh a) (weakenVars w gv) (k weakenId))) $
   mkMapF (addLHSToBIEnv (weakenBIEnv w env) elhs (k weakenId)) (weakenArrayInstr w exp2) (weaken w aOut)

mkMapF env (Evar (Var s idx)) aOut = Exec (TVar s (lookupBIEnv idx env)) $ aOut :>: ArgsNil
mkMapF _ _ _ = undefined
